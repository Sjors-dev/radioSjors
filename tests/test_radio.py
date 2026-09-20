"""Offline tests: everything that does not need ffmpeg, liquidsoap or a network.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import struct
import sys
import tempfile
import time
import unittest
import wave
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from airadio.config import Config
from airadio.db import Database
from airadio.discovery import LastFM
from airadio.downloader import Downloader, DownloadResult
from airadio.library import Library
from airadio.liquidsoap import annotate_uri, format_metadata, parse_metadata
from airadio.publisher import SitePublisher
from airadio.queueing import QueueManager, wav_duration
from airadio.tts import TTS
from airadio.util import dedupe_key, normalize, safe_filename
from airadio.weather import Weather
from airadio.brain.intent import _rules
from airadio.brain.llm import LLM, extract_json
from airadio.brain.planner import (Planner, _clean_exchange,
                                   _drop_consecutive_talk, _spoken_number,
                                   _template_weather)

BASE_CONFIG = yaml.safe_load((Path(__file__).resolve().parent.parent
                              / "config" / "config.yaml").read_text(encoding="utf-8"))


def make_wav(path: Path, seconds: float = 3.0, rate: int = 8000,
             tone: int = 0) -> None:
    """A real (tiny) wav file so mutagen reports a genuine duration.

    `tone` varies the samples so two different tracks are never byte-identical,
    which is what the content-hash de-duplication keys off.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", tone % 30000) * int(rate * seconds))


class RadioTestCase(unittest.TestCase):
    """Builds a throwaway station in a temp dir."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="airadio-test-"))
        data = dict(BASE_CONFIG)
        data["tts"] = dict(data["tts"], engine="none")
        data["paths"] = {"library": "library", "queue": "queue",
                         "state": "state", "logs": "logs"}
        (self.tmp / "config").mkdir(parents=True, exist_ok=True)
        (self.tmp / "config" / "config.yaml").write_text(
            yaml.safe_dump(data), encoding="utf-8")

        self.cfg = Config.load(self.tmp / "config" / "config.yaml", root=self.tmp)
        self.db = Database(self.cfg.db_path)
        self.db.init()
        self.library = Library(self.db, self.cfg.path("library"))
        self.tts = TTS(self.cfg)
        self.queue = QueueManager(self.cfg, self.db, self.library, self.tts)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def seed_library(self, artists: int = 6, per_artist: int = 5) -> None:
        for a in range(artists):
            for t in range(per_artist):
                make_wav(self.cfg.path("library") / f"Artist {a} - Track {t}.wav",
                         seconds=3.0 + t * 0.5, tone=a * 100 + t + 1)
        self.library.scan()


# -- pure helpers -----------------------------------------------------------


class TestUtil(unittest.TestCase):
    def test_normalize_strips_decorations(self):
        self.assertEqual(normalize("Hyperballad (Official Video)"), "hyperballad")
        self.assertEqual(normalize("Let It Be [Remastered 2009]"), "let it be")

    def test_dedupe_key_matches_variants(self):
        self.assertEqual(
            dedupe_key("Bjork", "Hyperballad"),
            dedupe_key("Björk", "Hyperballad (Official Video)"),
        )

    def test_stylised_spellings_match_plain_ones(self):
        # The same act turns up both ways depending on the source.
        self.assertEqual(normalize("A$AP Rocky"), normalize("ASAP Rocky"))
        self.assertEqual(normalize("Joey Bada$$"), normalize("Joey Badass"))
        self.assertEqual(dedupe_key("A$AP Rocky", "L$D"),
                         dedupe_key("ASAP Rocky", "LSD"))

    def test_dedupe_key_separates_different_tracks(self):
        self.assertNotEqual(dedupe_key("Queen", "One Vision"),
                            dedupe_key("Queen", "One Year of Love"))

    def test_safe_filename(self):
        self.assertEqual(safe_filename('AC/DC: Back in Black?'),
                         "ACDC Back in Black")


class TestJsonExtraction(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"a": 2}\n```'), {"a": 2})

    def test_json_with_chatter_around_it(self):
        self.assertEqual(
            extract_json('Sure! Here you go:\n{"items": []}\nHope that helps.'),
            {"items": []})

    def test_garbage_returns_none(self):
        self.assertIsNone(extract_json("no json here at all"))


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


GEMINI_OK = {"candidates": [{"content": {"parts": [{"text": "from gemini"}]}}]}
GROQ_OK = {"choices": [{"message": {"content": "from groq"}}]}


class TestLLMFailover(unittest.TestCase):
    """The failover path, which cannot be exercised against the live free tiers."""

    def setUp(self):
        import os

        self._saved = {k: os.environ.get(k) for k in ("GEMINI_API_KEY", "GROQ_API_KEY")}
        os.environ["GEMINI_API_KEY"] = "gem-test"
        os.environ["GROQ_API_KEY"] = "groq-test"
        data = dict(BASE_CONFIG)
        data["llm"] = dict(data["llm"], max_retries=0)
        self.llm = LLM(Config(data, Path(".")))
        self.calls: list[str] = []

    def tearDown(self):
        import os

        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _responder(self, gemini, groq):
        def post(url, **kwargs):
            if "googleapis" in url:
                self.calls.append("gemini")
                return gemini
            self.calls.append("groq")
            return groq
        return post

    def test_gemini_answers_first(self):
        self.llm._session.post = self._responder(
            FakeResponse(200, GEMINI_OK), FakeResponse(200, GROQ_OK))
        self.assertEqual(self.llm.complete("s", "u"), "from gemini")
        self.assertEqual(self.calls, ["gemini"])

    def test_rate_limited_gemini_falls_through_to_groq(self):
        self.llm._session.post = self._responder(
            FakeResponse(429, {"error": "quota"}), FakeResponse(200, GROQ_OK))
        self.assertEqual(self.llm.complete("s", "u"), "from groq")
        self.assertEqual(self.calls, ["gemini", "groq"])

    def test_rate_limited_provider_is_skipped_next_time(self):
        self.llm._session.post = self._responder(
            FakeResponse(429, {"error": "quota"}), FakeResponse(200, GROQ_OK))
        self.llm.complete("s", "u")
        self.calls.clear()
        self.llm.complete("s", "u")
        self.assertEqual(self.calls, ["groq"], "cooling-down provider was retried")

    def test_server_error_falls_through(self):
        self.llm._session.post = self._responder(
            FakeResponse(503, {"error": "down"}), FakeResponse(200, GROQ_OK))
        self.assertEqual(self.llm.complete("s", "u"), "from groq")

    def test_both_down_raises_so_the_caller_can_degrade(self):
        from airadio.brain.llm import LLMUnavailable

        self.llm._session.post = self._responder(
            FakeResponse(500, {}), FakeResponse(500, {}))
        with self.assertRaises(LLMUnavailable):
            self.llm.complete("s", "u")

    def test_complete_json_parses_a_fenced_reply(self):
        payload = {"candidates": [{"content": {"parts":
                  [{"text": '```json\n{"items": [1, 2]}\n```'}]}}]}
        self.llm._session.post = self._responder(
            FakeResponse(200, payload), FakeResponse(200, GROQ_OK))
        self.assertEqual(self.llm.complete_json("s", "u"), {"items": [1, 2]})

    def test_budgets_come_from_config(self):
        data = dict(BASE_CONFIG)
        data["llm"] = dict(data["llm"], max_tokens={"plan": 12345})
        llm = LLM(Config(data, Path(".")))
        self.assertEqual(llm.budget("plan"), 12345)
        self.assertEqual(llm.budget("intent"), 2048, "defaults should survive")
        self.assertEqual(llm.budget("nonsense"), 4096, "unknown job needs a floor")

    def test_bad_budget_value_is_ignored_not_fatal(self):
        data = dict(BASE_CONFIG)
        data["llm"] = dict(data["llm"], max_tokens={"plan": "lots"})
        self.assertEqual(LLM(Config(data, Path("."))).budget("plan"), 8000)

    def test_budget_is_actually_sent_to_the_provider(self):
        sent = {}

        def post(url, **kwargs):
            sent.update(kwargs.get("json") or {})
            return FakeResponse(200, GROQ_OK)

        data = dict(BASE_CONFIG)
        data["llm"] = dict(data["llm"], providers=["groq"], max_tokens={"plan": 7777})
        llm = LLM(Config(data, Path(".")))
        llm._session.post = post
        llm.complete("s", "u", max_tokens=llm.budget("plan"))
        self.assertEqual(sent.get("max_tokens"), 7777)

    def test_empty_reply_reports_the_finish_reason(self):
        from airadio.brain.llm import LLMUnavailable

        truncated = {"candidates": [{"content": {"parts": [{"text": ""}]},
                                     "finishReason": "MAX_TOKENS"}]}
        self.llm._session.post = self._responder(
            FakeResponse(200, truncated), FakeResponse(500, {}))
        with self.assertRaises(LLMUnavailable) as caught:
            self.llm.complete("s", "u")
        self.assertIn("MAX_TOKENS", str(caught.exception))

    def test_no_keys_means_disabled(self):
        import os

        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GROQ_API_KEY", None)
        self.assertFalse(LLM(Config(BASE_CONFIG, Path("."))).enabled)


class TestMetadataParsing(unittest.TestCase):
    def test_parses_the_newest_block_which_is_the_last(self):
        # Liquidsoap prints metadata in the order it went to air, so block 1 is
        # the OLDEST. Reading the first block pinned now-playing to whatever
        # was on when the stream started, and it never moved again.
        response = ('--- 1 ---\nartist="Old"\ntitle="Older"\n'
                    '--- 2 ---\nartist="Khruangbin"\ntitle="August 10"')
        self.assertEqual(format_metadata(parse_metadata(response)),
                         "Khruangbin - August 10")

    def test_single_block(self):
        response = '--- 1 ---\nartist="Gunna"\ntitle="fukumean"'
        self.assertEqual(format_metadata(parse_metadata(response)),
                         "Gunna - fukumean")

    def test_no_metadata_yet(self):
        self.assertEqual(format_metadata(parse_metadata("")), "")

    def test_falls_back_to_filename(self):
        meta = parse_metadata('--- 1 ---\nfilename="/music/Foo - Bar.mp3"')
        self.assertEqual(format_metadata(meta), "Foo - Bar")

    def test_patter_gets_a_hard_cut(self):
        uri = annotate_uri("/queue/p.wav", title="Station ID", artist="Radio",
                           no_crossfade=True)
        self.assertIn('liq_cross_duration="0."', uri)
        self.assertIn('liq_fade_in="0."', uri)
        self.assertIn('liq_fade_out="0."', uri)

    def test_music_keeps_its_crossfade(self):
        uri = annotate_uri("/music/x.mp3", title="T", artist="A")
        self.assertNotIn("liq_cross", uri)

    def test_annotate_uri_escapes_quotes_and_colons(self):
        uri = annotate_uri("/music/x.mp3", title='He said "hi": ok', artist="A")
        self.assertTrue(uri.startswith("annotate:"))
        self.assertIn("title=\"He said 'hi' - ok\"", uri)
        self.assertTrue(uri.endswith("x.mp3"))


class FakeLiquidsoap:
    """A socket server that answers like liquidsoap's telnet, so the batching
    and the END-framing are tested against real bytes rather than a mock."""

    def __init__(self, replies: dict[str, str]):
        import socket as socket_module
        import threading

        self.replies = replies
        self.commands_seen: list[str] = []
        self.said_goodbye = False
        self.connections = 0
        self._server = socket_module.socket()
        self._server.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(4)
        self.port = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            self.connections += 1
            with conn:
                buffer = ""
                while True:
                    try:
                        data = conn.recv(4096)
                    except OSError:
                        break
                    if not data:
                        break
                    buffer += data.decode("utf-8", "replace").replace("\r", "")
                    while "\n" in buffer:
                        line, _, buffer = buffer.partition("\n")
                        line = line.strip()
                        if line == "quit":
                            self.said_goodbye = True
                            return
                        self.commands_seen.append(line)
                        body = self.replies.get(line, "")
                        payload = (body + "\n" if body else "") + "END\r\n"
                        conn.sendall(payload.encode("utf-8"))

    def close(self):
        try:
            self._server.close()
        except OSError:
            pass


class TestLiquidsoapProtocol(unittest.TestCase):
    def setUp(self):
        from airadio.liquidsoap import LiquidsoapClient

        self.server = FakeLiquidsoap({
            "uptime": "5 days",
            "aiqueue.queue": "3 4 5",
            "reqqueue.queue": "",
            "caster.metadata": '--- 1 ---\nartist="Gunna"\ntitle="fukumean"',
            "aiqueue.push /music/x.mp3": "7",
        })
        self.client = LiquidsoapClient("127.0.0.1", self.server.port, timeout=5.0)

    def tearDown(self):
        self.server.close()

    def test_single_command(self):
        self.assertEqual(self.client.command("uptime"), "5 days")

    def test_batched_commands_come_back_in_order(self):
        responses = self.client.commands(["uptime", "aiqueue.queue"])
        self.assertEqual(responses[0], "5 days")
        self.assertEqual(responses[1], "3 4 5")

    def test_batch_uses_one_connection(self):
        self.client.commands(["uptime", "aiqueue.queue", "caster.metadata"])
        self.assertEqual(self.server.connections, 1)

    def test_client_says_goodbye(self):
        # Otherwise liquidsoap logs "disconnected without saying goodbye" on
        # every poll, which is every few seconds, forever.
        self.client.command("uptime")
        self.assertTrue(self.server.said_goodbye)

    def test_poll_returns_everything_the_feeder_needs(self):
        state = self.client.poll("aiqueue")
        self.assertIsNotNone(state)
        self.assertEqual(state["now_playing"], "Gunna - fukumean")
        self.assertEqual(state["depths"]["aiqueue"], 3)

    def test_poll_reads_both_queues_in_one_connection(self):
        state = self.client.poll("aiqueue", "reqqueue")
        self.assertEqual(state["depths"]["aiqueue"], 3)
        self.assertEqual(state["depths"]["reqqueue"], 0)
        self.assertEqual(self.server.connections, 1)

    def test_poll_on_a_dead_server_returns_none(self):
        from airadio.liquidsoap import LiquidsoapClient

        dead = LiquidsoapClient("127.0.0.1", 9, timeout=1.0)
        self.assertIsNone(dead.poll("aiqueue"))
        self.assertFalse(dead.connected)

    def test_push(self):
        self.assertTrue(self.client.push("aiqueue", "/music/x.mp3"))


class TestIntentRules(unittest.TestCase):
    def test_play_with_artist(self):
        result = _rules("play bohemian rhapsody by queen")
        self.assertEqual(result["kind"], "track")
        self.assertEqual(result["artist"].lower(), "queen")
        self.assertEqual(result["title"].lower(), "bohemian rhapsody")

    def test_play_without_artist(self):
        result = _rules("play teardrop")
        self.assertEqual(result["kind"], "track")
        self.assertEqual(result["artist"], "")

    def test_vibe_shift(self):
        self.assertEqual(_rules("make it darker")["kind"], "vibe")

    def test_skip_is_not_a_ban(self):
        # Skipping is harmless, banning throws a track out of the station.
        # Reading one as the other is the worst mistake this classifier can
        # make, so it gets its own test.
        for phrase in ("skip", "!skip", "skip this song please", "skip it",
                       "next song", "move on"):
            self.assertEqual(_rules(phrase)["kind"], "skip", phrase)

    def test_ban_is_still_a_ban(self):
        for phrase in ("delete this song", "never play this again",
                       "ban Runaway by Kanye West", "blacklist this"):
            self.assertEqual(_rules(phrase)["kind"], "ban", phrase)

    def test_skip_matches_whole_words_only(self):
        self.assertNotEqual(_rules("skipper")["kind"], "skip")
        self.assertEqual(_rules("what's next?")["kind"], "question")

    def test_track_named_next_is_still_a_request(self):
        result = _rules("play next to you by john legend")
        self.assertEqual(result["kind"], "track")
        self.assertEqual(result["artist"].lower(), "john legend")

    def test_question(self):
        self.assertEqual(_rules("what's playing?")["kind"], "question")


class TestConfig(unittest.TestCase):
    def test_mood_map_covers_every_hour(self):
        cfg = Config(BASE_CONFIG, Path("."))
        for hour in range(24):
            profile = cfg.mood_for_hour(hour)
            self.assertNotEqual(profile["name"], "default",
                                f"hour {hour} is not covered by the mood map")

    def test_default_genres_still_apply_after_a_local_override_replaces_mood_map(self):
        # The actual bug: config.local.yaml sets its own planner.mood_map to
        # customise the mood text, written before genres:/default_genres
        # existed. Since a list overrides wholesale (see _deep_merge), any
        # genres: living directly on the old entries would be gone with no
        # error -- default_genres is a dict, keyed by the same slot names, so
        # it must still apply even though this mood_map has no genres at all.
        data = dict(BASE_CONFIG)
        data["planner"] = dict(data["planner"])
        data["planner"]["mood_map"] = [
            {"hours": [0, 24], "name": "afternoon", "mood": "my own words",
             "energy": [3, 4]},
        ]
        cfg = Config(data, Path("."))
        profile = cfg.mood_for_hour(14)
        self.assertEqual(profile["mood"], "my own words")
        self.assertEqual(profile["genres"],
                         BASE_CONFIG["planner"]["default_genres"]["afternoon"])

    def test_an_entrys_own_genres_win_over_the_default(self):
        data = dict(BASE_CONFIG)
        data["planner"] = dict(data["planner"])
        data["planner"]["mood_map"] = [
            {"hours": [0, 24], "name": "afternoon", "mood": "x",
             "energy": [1, 5], "genres": ["metal"]},
        ]
        cfg = Config(data, Path("."))
        self.assertEqual(cfg.mood_for_hour(14)["genres"], ["metal"])

    def test_an_unrecognised_slot_name_gets_no_genres_rather_than_crashing(self):
        data = dict(BASE_CONFIG)
        data["planner"] = dict(data["planner"])
        data["planner"]["mood_map"] = [
            {"hours": [0, 24], "name": "brand new slot", "mood": "x",
             "energy": [1, 5]},
        ]
        cfg = Config(data, Path("."))
        self.assertNotIn("genres", cfg.mood_for_hour(14))


# -- downloader quality filters ---------------------------------------------


class TestQualityFilters(RadioTestCase):
    def setUp(self):
        super().setUp()
        self.downloader = Downloader(self.cfg, self.db, self.library, LastFM(""))

    def candidate(self, title: str, channel: str = "Some Channel",
                  duration: float = 240.0) -> dict:
        return {"id": "x", "title": title, "channel": channel, "duration": duration}

    def test_rejects_live_version(self):
        score, reason = self.downloader.score_candidate(
            self.candidate("Song Name (Live at Wembley)"), "Band", "Song Name", 240)
        self.assertLess(score, 0)
        self.assertIn("live", reason)

    def test_rejects_hour_long_loop(self):
        score, _ = self.downloader.score_candidate(
            self.candidate("Song Name [1 hour loop]", duration=3600), "Band",
            "Song Name", 240)
        self.assertLess(score, 0)

    def test_rejects_wrong_duration(self):
        score, reason = self.downloader.score_candidate(
            self.candidate("Song Name", duration=600), "Band", "Song Name", 240)
        self.assertLess(score, 0)
        self.assertIn("off the expected", reason)

    def test_accepts_topic_channel(self):
        score, reason = self.downloader.score_candidate(
            self.candidate("Song Name", channel="Band - Topic", duration=242),
            "Band", "Song Name", 240)
        self.assertEqual(reason, "ok")
        self.assertGreater(score, 60)

    def test_topic_channel_outranks_random_upload(self):
        official, _ = self.downloader.score_candidate(
            self.candidate("Song Name", channel="Band - Topic", duration=240),
            "Band", "Song Name", 240)
        random_upload, _ = self.downloader.score_candidate(
            self.candidate("Song Name", channel="mixesdaily", duration=240),
            "Band", "Song Name", 240)
        self.assertGreater(official, random_upload)

    def test_requested_live_track_is_not_rejected(self):
        # "Live Forever" must survive the "live" reject pattern.
        score, reason = self.downloader.score_candidate(
            self.candidate("Live Forever", channel="Oasis - Topic", duration=286),
            "Oasis", "Live Forever", 286)
        self.assertEqual(reason, "ok")
        self.assertGreater(score, 0)

    def test_allow_remix_relaxes_the_filter(self):
        blocked, _ = self.downloader.score_candidate(
            self.candidate("Song Name (Remix)"), "Band", "Song Name", 240)
        allowed, reason = self.downloader.score_candidate(
            self.candidate("Song Name (Remix)"), "Band", "Song Name", 240,
            allow_remix=True)
        self.assertLess(blocked, 0)
        self.assertEqual(reason, "ok")
        self.assertGreaterEqual(allowed, 0)

    def test_reject_patterns_match_whole_words_only(self):
        # Substring matching used to reject these: "live" inside "Livewire",
        # "cover" inside "Coverdale", "mix" inside "Remix".
        for title, artist in (("Livewire", "Motley Crue"),
                              ("Coverdale Page", "Coverdale")):
            score, reason = self.downloader.score_candidate(
                self.candidate(title, channel=f"{artist} - Topic", duration=240),
                artist, title, 240)
            self.assertEqual(reason, "ok", f"{title} was wrongly rejected")

    def test_remix_is_still_rejected_by_default(self):
        score, reason = self.downloader.score_candidate(
            self.candidate("Song Name (Nightcore Remix)"), "Band", "Song Name", 240)
        self.assertLess(score, 0)

    def test_rejects_missing_duration(self):
        score, reason = self.downloader.score_candidate(
            self.candidate("Song Name", duration=None), "Band", "Song Name", 240)
        self.assertLess(score, 0)
        self.assertIn("duration", reason)


# -- library ----------------------------------------------------------------


class FakeLastFM(LastFM):
    """A Last.fm whose similar-artist graph and top tracks are deterministic."""

    def __init__(self, graph: dict[str, list[str]], tracks_per_artist: int = 20):
        super().__init__("fake-key")
        self.graph = graph
        self.tracks_per_artist = tracks_per_artist
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return True

    def similar_artists(self, artist, limit=12):
        return self.graph.get(artist, [])[:limit]

    def top_tracks(self, artist, limit=15):
        self.calls += 1
        return [{"artist": artist, "title": f"Track {n}"}
                for n in range(min(limit, self.tracks_per_artist))]

    def track_info(self, artist, title):
        return {"duration": 200.0, "tags": [], "listeners": 1000,
                "artist": artist, "title": title}

    def artist_tags(self, artist):
        return ["test"]


GRAPH = {
    "Kanye West": ["Jay-Z", "Kid Cudi", "Pusha T"],
    "Radiohead": ["Blur", "Muse", "Portishead"],
    "Billie Holiday": ["Ella Fitzgerald", "Nina Simone", "Sarah Vaughan"],
    "Gunna": ["Young Thug", "Lil Baby", "Future"],
}


class TestDiscoveryBreadth(RadioTestCase):
    """Discovery must widen the library, not deepen a handful of artists.

    The first build took eight top tracks per artist and bailed after two
    seeds, so a whole night of downloading produced 77 tracks across 14
    artists while most of the configured seeds were never touched.
    """

    def setUp(self):
        super().setUp()
        self.lastfm = FakeLastFM(GRAPH)
        self.downloader = Downloader(self.cfg, self.db, self.library, self.lastfm)

    def counts(self, tracks):
        out: dict[str, int] = {}
        for track in tracks:
            out[track["artist"]] = out.get(track["artist"], 0) + 1
        return out

    def test_no_artist_dominates_the_pool(self):
        pool = self.lastfm.expand(list(GRAPH), want=30, per_artist=3)
        counts = self.counts(pool)
        self.assertLessEqual(max(counts.values()), 3, counts)

    def test_pool_spans_many_artists(self):
        pool = self.lastfm.expand(list(GRAPH), want=30, per_artist=3)
        self.assertGreaterEqual(len(self.counts(pool)), 10)

    def test_every_seed_is_reachable(self):
        # Not every seed in one pass, but across several passes all of them
        # should appear -- the old version explored one or two and stopped.
        seen: set[str] = set()
        for _ in range(8):
            seen.update(t["artist"] for t in
                        self.lastfm.expand(list(GRAPH), want=30, per_artist=3))
        for seed in GRAPH:
            self.assertIn(seed, seen, f"{seed} was never explored")

    def test_download_order_alternates_artists(self):
        # Round-robin matters: a FIFO candidate queue downloads in insertion
        # order, so grouping by artist means hours of one discography.
        pool = self.lastfm.expand(list(GRAPH), want=24, per_artist=3)
        first_pass = [t["artist"] for t in pool[:8]]
        self.assertEqual(len(set(first_pass)), len(first_pass),
                         f"first downloads repeat an artist: {first_pass}")

    def add_tracks(self, artist: str, count: int) -> None:
        for n in range(count):
            self.db.execute(
                "INSERT INTO tracks(path, dedupe_key, title, artist, added_at) "
                "VALUES(?,?,?,?,?)",
                (f"/x/{artist}-{n}.mp3", f"{normalize(artist)}|track {n}",
                 f"Track {n}", artist, 0))

    def test_seed_artists_get_a_higher_ceiling(self):
        # The cap exists to stop one name crowding everything out, not to
        # ration the artists the listener explicitly asked for.
        self.cfg._data["discovery"]["seed_artists"] = ["Kanye West"]
        self.cfg._data["discovery"]["max_tracks_per_artist"] = 5
        self.cfg._data["discovery"]["max_tracks_per_seed_artist"] = 30
        self.assertEqual(self.downloader.artist_cap("Kanye West"), 30)
        self.assertEqual(self.downloader.artist_cap("Some Neighbour"), 5)

    def test_seed_cap_survives_spelling_differences(self):
        self.cfg._data["discovery"]["seed_artists"] = ["A$AP Rocky"]
        self.cfg._data["discovery"]["max_tracks_per_artist"] = 5
        self.cfg._data["discovery"]["max_tracks_per_seed_artist"] = 30
        self.assertEqual(self.downloader.artist_cap("ASAP Rocky"), 30)

    def test_a_favourite_keeps_growing_past_the_discovered_cap(self):
        self.cfg._data["discovery"]["seed_artists"] = ["Kanye West"]
        self.cfg._data["discovery"]["max_tracks_per_artist"] = 5
        self.cfg._data["discovery"]["max_tracks_per_seed_artist"] = 30
        self.add_tracks("Kanye West", 12)
        self.add_tracks("Some Neighbour", 6)
        self.assertFalse(self.downloader.artist_is_full("Kanye West"))
        self.assertTrue(self.downloader.artist_is_full("Some Neighbour"))

    def test_zero_means_no_ceiling(self):
        self.cfg._data["discovery"]["max_tracks_per_artist"] = 0
        self.add_tracks("Nobody", 50)
        self.assertFalse(self.downloader.artist_is_full("Nobody"))

    def test_lock_contention_does_not_burn_an_attempt(self):
        # Losing the download lock to the brain is not a failure of the track.
        # Counting it marked good candidates as permanently rejected after
        # three unlucky races.
        self.downloader.lock_path.write_text("999999", encoding="utf-8")
        self.db.execute(
            "INSERT INTO candidates(dedupe_key, artist, title, source, created_at) "
            "VALUES(?,?,?,?,?)", ("a|b", "Artist", "Title", "test", 0))

        for _ in range(5):
            result = self.downloader.fetch_next_candidate()
            self.assertTrue(result.busy, result)

        row = self.db.one("SELECT attempts, status FROM candidates WHERE artist='Artist'")
        self.assertEqual(row["attempts"], 0)
        self.assertEqual(row["status"], "new")

    def test_lock_from_a_dead_process_is_cleared(self):
        # A download killed mid-flight (Ctrl-C, pkill, reboot) never releases
        # its lock. Waiting out the 30 minute staleness timeout behind a
        # process that no longer exists blocked a whole bootstrap run.
        self.downloader.lock_path.write_text("999999", encoding="utf-8")
        self.downloader._lock_holder_alive = lambda: False
        self.assertTrue(self.downloader._acquire_lock())

    def test_lock_from_a_live_process_is_respected(self):
        self.downloader.lock_path.write_text("999999", encoding="utf-8")
        self.downloader._lock_holder_alive = lambda: True
        self.assertFalse(self.downloader._acquire_lock())

    def test_unknown_holder_falls_back_to_the_age_check(self):
        self.downloader.lock_path.write_text("999999", encoding="utf-8")
        self.downloader._lock_holder_alive = lambda: None
        self.assertFalse(self.downloader._acquire_lock(), "a fresh lock holds")

        import os as os_module
        old = time.time() - 4000
        os_module.utime(self.downloader.lock_path, (old, old))
        self.assertTrue(self.downloader._acquire_lock(), "an old lock expires")

    def test_corrupt_lock_file_does_not_block_forever(self):
        self.downloader.lock_path.write_text("not a pid", encoding="utf-8")
        self.assertFalse(self.downloader._lock_holder_alive())
        self.assertTrue(self.downloader._acquire_lock())

    def test_our_own_stale_lock_is_reclaimed(self):
        import os as os_module
        self.downloader.lock_path.write_text(str(os_module.getpid()),
                                             encoding="utf-8")
        self.assertTrue(self.downloader._acquire_lock())

    def test_lock_contention_is_flagged_busy_not_merely_failed(self):
        self.downloader.lock_path.write_text("999999", encoding="utf-8")
        result = self.downloader.fetch("Artist", "Title")
        self.assertFalse(result.ok)
        self.assertTrue(result.busy)

    def test_junk_candidate_is_rejected_on_the_first_try(self):
        # Last.fm returns non-music entries. If every search result fails the
        # quality checks, running the identical search twice more just wastes
        # yt-dlp calls on a two-core CPU.
        self.db.execute(
            "INSERT INTO candidates(dedupe_key, artist, title, source, created_at) "
            "VALUES(?,?,?,?,?)", ("j|k", "The Throne of Allah", "Mindblowing",
                                  "lastfm", 0))
        self.downloader._search = lambda artist, title: [
            {"id": "x", "url": "u", "title": "A lecture", "channel": "Some Channel",
             "duration": 4597},
        ]
        result = self.downloader.fetch_next_candidate()
        self.assertFalse(result.ok)
        self.assertFalse(result.retryable)
        row = self.db.one(
            "SELECT status FROM candidates WHERE artist='The Throne of Allah'")
        self.assertEqual(row["status"], "rejected")

    def test_a_search_that_returned_nothing_is_retried(self):
        # An empty search can be a transient network problem, unlike results
        # that are all junk.
        self.db.execute(
            "INSERT INTO candidates(dedupe_key, artist, title, source, created_at) "
            "VALUES(?,?,?,?,?)", ("e|f", "Flaky Artist", "Flaky", "lastfm", 0))
        self.downloader._search = lambda artist, title: []
        result = self.downloader.fetch_next_candidate()
        self.assertTrue(result.retryable)
        row = self.db.one("SELECT status FROM candidates WHERE artist='Flaky Artist'")
        self.assertEqual(row["status"], "new")

    def test_a_real_failure_still_counts(self):
        # No lock held; Last.fm is fake and yt-dlp will find nothing.
        self.db.execute(
            "INSERT INTO candidates(dedupe_key, artist, title, source, created_at) "
            "VALUES(?,?,?,?,?)", ("c|d", "Nonexistent Artist", "Nonexistent", "test", 0))
        self.downloader._search = lambda artist, title: []
        result = self.downloader.fetch_next_candidate()
        self.assertFalse(result.busy)
        row = self.db.one(
            "SELECT attempts FROM candidates WHERE artist='Nonexistent Artist'")
        self.assertEqual(row["attempts"], 1)

    def test_artist_cap_blocks_further_candidates(self):
        self.cfg._data["discovery"]["seed_artists"] = []
        self.cfg._data["discovery"]["max_tracks_per_artist"] = 2
        for n in range(3):
            self.db.execute(
                "INSERT INTO tracks(path, dedupe_key, title, artist, added_at) "
                "VALUES(?,?,?,?,?)",
                (f"/x/kanye{n}.mp3", f"kanye west|track {n}", f"Track {n}",
                 "Kanye West", 0))
        self.downloader.seed_candidates(wanted=40)
        rows = self.db.query(
            "SELECT artist FROM candidates WHERE artist='Kanye West'")
        self.assertEqual(rows, [], "queued more of an artist already at the cap")

    def test_seeding_favours_under_represented_artists(self):
        # Seeding from the most-played artists is a feedback loop; the ones
        # that already dominate pull in more of themselves.
        for n in range(9):
            self.db.execute(
                "INSERT INTO tracks(path, dedupe_key, title, artist, added_at) "
                "VALUES(?,?,?,?,?)",
                (f"/x/g{n}.mp3", f"gunna|track {n}", f"Track {n}", "Gunna", 0))
        self.db.execute(
            "INSERT INTO tracks(path, dedupe_key, title, artist, added_at) "
            "VALUES(?,?,?,?,?)",
            ("/x/r1.mp3", "radiohead|creep", "Creep", "Radiohead", 0))
        rows = self.db.query(
            "SELECT artist, COUNT(*) AS n FROM tracks WHERE missing=0 "
            "GROUP BY artist ORDER BY n ASC LIMIT 10")
        self.assertEqual(rows[0]["artist"], "Radiohead")


class TestBackfillOnDemand(RadioTestCase):
    """A chat request naming a thin artist should fetch more of them, not
    just hand back the one track already on hand."""

    def setUp(self):
        super().setUp()
        self.lastfm = FakeLastFM({}, tracks_per_artist=20)
        self.downloader = Downloader(self.cfg, self.db, self.library, self.lastfm)
        self.fetched: list[tuple[str, str]] = []

    def add_tracks(self, artist: str, count: int) -> None:
        for n in range(count):
            self.db.execute(
                "INSERT INTO tracks(path, dedupe_key, title, artist, added_at) "
                "VALUES(?,?,?,?,?)",
                (f"/x/{normalize(artist)}-{n}.mp3",
                 f"{normalize(artist)}|track {n}", f"Track {n}", artist, 0))

    def stub_fetch_always_succeeds(self) -> None:
        """Stand in for the real fetch(): records the call and actually
        inserts a row, so artist_track_count reflects progress exactly like
        the real downloader would."""
        def fake_fetch(artist, title, allow_remix=False, max_seconds=None):
            self.fetched.append((artist, title))
            self.add_tracks_one(artist, title)
            return DownloadResult(True, {"artist": artist, "title": title})
        self.downloader.fetch = fake_fetch

    def add_tracks_one(self, artist: str, title: str) -> None:
        self.db.execute(
            "INSERT INTO tracks(path, dedupe_key, title, artist, added_at) "
            "VALUES(?,?,?,?,?)",
            (f"/x/{normalize(artist)}-{normalize(title)}.mp3",
             dedupe_key(artist, title), title, artist, 0))

    def test_tops_up_to_the_minimum(self):
        self.add_tracks("Thin Artist", 1)
        self.stub_fetch_always_succeeds()
        added = self.downloader.backfill_artist("Thin Artist", minimum=4)
        self.assertEqual(added, 3)
        self.assertEqual(self.downloader.artist_track_count("Thin Artist"), 4)

    def test_does_nothing_once_the_minimum_is_already_met(self):
        self.add_tracks("Well Stocked", 5)
        self.stub_fetch_always_succeeds()
        added = self.downloader.backfill_artist("Well Stocked", minimum=4)
        self.assertEqual(added, 0)
        self.assertEqual(self.fetched, [])

    def test_never_fetches_past_the_artists_own_cap(self):
        # Non-seed artists are capped at discovery.max_tracks_per_artist (8
        # in the shipped config). Asking for way more than that must not
        # blow past the cap just because the listener asked for a lot.
        self.assertEqual(self.cfg.get("discovery.max_tracks_per_artist"), 8)
        self.stub_fetch_always_succeeds()
        added = self.downloader.backfill_artist("New Artist", minimum=50)
        self.assertEqual(added, 8)
        self.assertEqual(self.downloader.artist_track_count("New Artist"), 8)

    def test_seed_artists_get_the_higher_ceiling(self):
        self.cfg._data["discovery"]["seed_artists"] = ["Favourite"]
        self.lastfm.tracks_per_artist = 50  # more than either ceiling
        self.stub_fetch_always_succeeds()
        added = self.downloader.backfill_artist("Favourite", minimum=50)
        self.assertEqual(added, 30)  # max_tracks_per_seed_artist default

    def test_skips_candidates_already_owned_or_banned(self):
        self.add_tracks("Mixed Bag", 1)  # "Track 0" already present
        self.library.ban({"id": 1, "artist": "Mixed Bag", "title": "Track 1",
                          "path": "/x/mixed-1.mp3", "dedupe_key":
                          dedupe_key("Mixed Bag", "Track 1")},
                         reason="test", banned_dir=self.cfg.path("banned"))
        self.stub_fetch_always_succeeds()
        self.downloader.backfill_artist("Mixed Bag", minimum=4)
        # Track 0 (owned) and Track 1 (banned) must never reach fetch().
        self.assertNotIn(("Mixed Bag", "Track 0"), self.fetched)
        self.assertNotIn(("Mixed Bag", "Track 1"), self.fetched)

    def test_stops_at_the_time_budget_rather_than_finishing_the_list(self):
        self.stub_fetch_always_succeeds()
        added = self.downloader.backfill_artist("Slow Artist", minimum=10,
                                                 budget_seconds=-1)
        self.assertEqual(added, 0)
        self.assertEqual(self.fetched, [])

    def test_stops_when_the_download_lock_is_busy(self):
        calls = []
        def busy_fetch(artist, title, allow_remix=False, max_seconds=None):
            calls.append((artist, title))
            return DownloadResult(False, busy=True, reason="another download "
                                  "is running")
        self.downloader.fetch = busy_fetch
        added = self.downloader.backfill_artist("Busy Artist", minimum=4)
        self.assertEqual(added, 0)
        # One attempt, then it gives up rather than hammering a busy lock.
        self.assertEqual(len(calls), 1)

    def test_no_lastfm_key_means_no_backfill(self):
        downloader = Downloader(self.cfg, self.db, self.library, LastFM(""))
        added = downloader.backfill_artist("Anybody", minimum=4)
        self.assertEqual(added, 0)


class _StubLiquidsoap:
    """Stands in for the stream engine. No socket, ever."""

    connected = False

    def poll(self, *queue_ids):
        return None

    def push(self, *args, **kwargs):
        return False

    def skip(self):
        return False


class TestChatBackfillIntegration(RadioTestCase):
    """A chat request naming a thin artist, wired through a real Runner with
    the stream engine and the network stubbed out."""

    def setUp(self):
        super().setUp()
        from airadio.runner import Runner

        self.runner = Runner(self.cfg)
        self.runner.ls = _StubLiquidsoap()
        self.lastfm = FakeLastFM({}, tracks_per_artist=20)
        self.runner.lastfm = self.lastfm
        self.runner.downloader = Downloader(self.cfg, self.runner.db,
                                            self.runner.library, self.lastfm)
        self.fetched: list[tuple[str, str]] = []

        def fake_fetch(artist, title, allow_remix=False, max_seconds=None):
            self.fetched.append((artist, title))
            self._insert_track(artist, title)
            return DownloadResult(True, {"artist": artist, "title": title})

        self.runner.downloader.fetch = fake_fetch

    def _insert_track(self, artist: str, title: str) -> None:
        # plan_block filters candidates to files that actually exist on disk,
        # same as the real downloader leaves behind -- a DB-only row would be
        # invisible to the planner and the test would not prove anything.
        path = (self.runner.cfg.path("library")
               / f"{normalize(artist)}-{normalize(title)}.mp3")
        path.write_bytes(b"")
        self.runner.db.execute(
            "INSERT INTO tracks(path, dedupe_key, title, artist, duration, "
            "added_at) VALUES(?,?,?,?,?,?)",
            (str(path), dedupe_key(artist, title), title, artist, 180.0, 0))

    def add_tracks(self, artist: str, count: int) -> None:
        for n in range(count):
            self._insert_track(artist, f"Track {n}")

    def test_vibe_request_backfills_a_thin_named_artist(self):
        self.add_tracks("Westside Gunn", 1)
        reply = self.runner._handle_vibe_request(
            {"mood": "in the mood for some Westside Gunn"})
        self.assertGreaterEqual(
            self.runner.downloader.artist_track_count("Westside Gunn"), 4)
        self.assertIn("Westside Gunn", reply)
        self.assertIn("Grabbed", reply)

    def test_the_backfilled_tracks_are_available_before_the_replan(self):
        # The point of doing this before set_mood/plan_block, not after: a
        # block built from the old, thin pool would not use them at all.
        self.add_tracks("Westside Gunn", 1)
        self.runner._handle_vibe_request(
            {"mood": "in the mood for some Westside Gunn"})
        planned = self.runner.db.one(
            "SELECT COUNT(*) AS n FROM queue_items WHERE artist='Westside Gunn' "
            "AND status IN ('ready', 'pending')")
        self.assertGreater(planned["n"], 0)

    def test_vibe_request_does_not_touch_an_already_well_stocked_artist(self):
        self.add_tracks("Well Stocked", 6)
        self.runner._handle_vibe_request({"mood": "some Well Stocked please"})
        self.assertEqual(self.fetched, [])

    def test_vibe_request_naming_nobody_does_not_call_lastfm(self):
        self.runner._handle_vibe_request({"mood": "something darker and slower"})
        self.assertEqual(self.fetched, [])

    def test_artist_only_request_backfills_then_queues_one(self):
        self.add_tracks("Thin Guy", 1)
        reply = self.runner._handle_track_request(
            {"artist": "Thin Guy", "title": ""})
        self.assertIn("Thin Guy", reply)
        self.assertGreaterEqual(
            self.runner.downloader.artist_track_count("Thin Guy"), 4)
        queued = self.runner.db.one(
            "SELECT COUNT(*) AS n FROM queue_items WHERE tier='request'")
        self.assertEqual(queued["n"], 1)

    def test_artist_only_request_no_longer_settles_for_a_fuzzy_match(self):
        # Before this fix, an artist-only request fell through to the fuzzy
        # title search on just the artist name and could return whichever one
        # track happened to match best, without ever trying to fetch more.
        self.add_tracks("One Song Wonder", 1)
        self.runner._handle_track_request({"artist": "One Song Wonder", "title": ""})
        self.assertTrue(self.fetched, "no backfill was attempted at all")

    def test_artist_only_request_with_nothing_and_no_lastfm(self):
        self.runner.lastfm = LastFM("")
        self.runner.downloader = Downloader(self.cfg, self.runner.db,
                                            self.runner.library, self.runner.lastfm)
        reply = self.runner._handle_track_request(
            {"artist": "Total Stranger", "title": ""})
        self.assertIn("not set up", reply)


class TestRequeueCommand(RadioTestCase):
    """The !requeue Discord command: clear the AI queue and rebuild it now,
    wired through a real Runner with the stream engine stubbed out."""

    def setUp(self):
        super().setUp()
        from airadio.runner import Runner

        self.runner = Runner(self.cfg)
        self.runner.ls = _StubLiquidsoap()
        self.seed_library(artists=8, per_artist=4)

    def _build_initial_block(self) -> None:
        track = dict(self.runner.db.query("SELECT * FROM tracks LIMIT 1")[0])
        self.runner.queue.build_block({
            "items": [{"kind": "song", "track": track}],
            "source": "test", "mood_name": "night", "show_note": "",
        })

    def test_requeue_drops_ready_ai_items_and_rebuilds(self):
        self._build_initial_block()
        blocks_before = self.runner.db.one(
            "SELECT COUNT(*) AS n FROM blocks")["n"]

        reply = self.runner._handle_requeue_request()

        self.assertIn("Cleared", reply)
        blocks_after = self.runner.db.one(
            "SELECT COUNT(*) AS n FROM blocks")["n"]
        self.assertEqual(blocks_after, blocks_before + 1,
                         "a genuinely fresh block should have been planned, "
                         "not just left alone")
        fresh = self.runner.db.one(
            "SELECT COUNT(*) AS n FROM queue_items WHERE tier='ai' AND "
            "status IN ('ready', 'pending')")
        # The fallback planner fills a real hour from the 32 seeded tracks,
        # far more than the single placeholder song _build_initial_block
        # queued -- proof the old, thin block was replaced, not added to.
        self.assertGreater(fresh["n"], 1)

    def test_requeue_reports_how_many_it_dropped(self):
        self._build_initial_block()
        dropped_count = self.runner.db.one(
            "SELECT COUNT(*) AS n FROM queue_items WHERE tier='ai' AND "
            "status='ready'")["n"]
        reply = self.runner._handle_requeue_request()
        self.assertIn(f"Cleared {dropped_count}", reply)

    def test_requeue_never_touches_the_request_tier(self):
        track = dict(self.runner.db.query("SELECT * FROM tracks LIMIT 1")[0])
        self.runner.queue.enqueue_request(track)
        self._build_initial_block()

        self.runner._handle_requeue_request()

        still_queued = self.runner.db.one(
            "SELECT COUNT(*) AS n FROM queue_items WHERE tier='request' AND "
            "status='ready'")
        self.assertEqual(still_queued["n"], 1,
                         "a track the listener explicitly asked for must "
                         "survive a requeue")

    def test_requeue_does_not_change_the_mood(self):
        self.runner.planner.set_mood("late-night jazz")
        self.runner._handle_requeue_request()
        self.assertEqual(self.runner.planner.current_mood(), "late-night jazz")

    def test_requeue_on_an_empty_library_says_so_without_crashing(self):
        # Simulate an empty library without a second Runner/db, which would
        # just see the same tracks seed_library already put there.
        self.runner.db.execute("UPDATE tracks SET missing=1")
        reply = self.runner._handle_requeue_request()
        self.assertIn("safety playlist", reply)

    def test_handle_request_dispatches_requeue_without_classifying(self):
        # The whole point of setting kind='requeue' at insert time: this is
        # deterministic and unambiguous, so no LLM call should ever happen
        # for it, unlike free-text chat.
        import airadio.runner as runner_module
        original = runner_module.classify

        def boom(*args, **kwargs):
            raise AssertionError("classify() must not run for a "
                                 "pre-classified requeue request")

        runner_module.classify = boom
        try:
            reply = self.runner._handle_request(
                {"kind": "requeue", "text": "requeue the ai queue",
                 "user": "tester"})
        finally:
            runner_module.classify = original
        self.assertIn("Cleared", reply)


class TestLibrary(RadioTestCase):
    def test_scan_indexes_files(self):
        self.seed_library(artists=3, per_artist=4)
        self.assertEqual(self.library.count(), 12)

    def test_scan_is_idempotent(self):
        self.seed_library(artists=3, per_artist=4)
        self.library.scan()
        self.assertEqual(self.library.count(), 12)

    def test_duplicate_file_is_skipped(self):
        self.seed_library(artists=1, per_artist=1)
        source = next(self.cfg.path("library").glob("*.wav"))
        shutil.copyfile(source, source.with_name("Artist 0 - Track 0 (copy).wav"))
        self.library.scan()
        self.assertEqual(self.library.count(), 1)

    def test_missing_file_is_marked(self):
        self.seed_library(artists=1, per_artist=2)
        next(self.cfg.path("library").glob("*.wav")).unlink()
        self.library.scan()
        self.assertEqual(self.library.count(), 1)

    def test_loose_find(self):
        self.seed_library(artists=2, per_artist=2)
        self.assertIsNotNone(self.library.find("artist 1 track 0"))
        self.assertIsNone(self.library.find("something that is not there"))


# -- planner ----------------------------------------------------------------


class TestBanning(RadioTestCase):
    """Banning must actually remove the track from every tier, including the
    safety playlist, which reads the library folder rather than the database."""

    def setUp(self):
        super().setUp()
        self.seed_library(artists=4, per_artist=3)
        self.downloader = Downloader(self.cfg, self.db, self.library, LastFM(""))
        self.track = dict(self.db.one("SELECT * FROM tracks LIMIT 1"))

    def ban(self):
        return self.library.ban(self.track, reason="test",
                                banned_dir=self.cfg.path("banned"))

    def test_file_leaves_the_library_folder(self):
        original = Path(self.track["path"])
        moved = self.ban()
        self.assertFalse(original.exists(), "banned file is still in library/")
        self.assertTrue(Path(moved).exists(), "banned file was lost, not moved")
        self.assertEqual(Path(moved).parent, self.cfg.path("banned"))

    def test_track_is_no_longer_schedulable(self):
        self.ban()
        planner = Planner(self.cfg, self.db, self.library, LLM(self.cfg))
        ids = [t["id"] for t in planner.select_candidates([1, 5], 50)]
        self.assertNotIn(self.track["id"], ids)

    def test_library_count_drops(self):
        before = self.library.count()
        self.ban()
        self.assertEqual(self.library.count(), before - 1)

    def test_is_banned_matches_spelling_variants(self):
        self.ban()
        self.assertTrue(self.library.is_banned(
            self.track["artist"], self.track["title"] + " (Official Video)"))

    def test_download_refuses_a_banned_track(self):
        self.ban()
        result = self.downloader.fetch(self.track["artist"], self.track["title"])
        self.assertFalse(result.ok)
        self.assertIn("banned", result.reason)

    def test_scan_does_not_readopt_a_banned_file(self):
        moved = self.ban()
        # Simulate the file being put back into the library by hand.
        shutil.copyfile(moved, self.cfg.path("library") / Path(moved).name)
        self.library.scan()
        self.assertFalse(self.library.has_track(
            self.track["artist"], self.track["title"]))

    def test_unban_restores_the_file_and_the_track(self):
        self.ban()
        key = dedupe_key(self.track["artist"], self.track["title"])
        self.assertTrue(self.library.unban(key, library_dir=self.cfg.path("library")))
        self.library.scan()
        self.assertIsNotNone(self.library.has_track(
            self.track["artist"], self.track["title"]))

    def test_unban_of_something_not_banned_is_a_no_op(self):
        self.assertFalse(self.library.unban("nope|nothing"))

    def test_banning_twice_does_not_error(self):
        self.ban()
        self.ban()
        row = self.db.one("SELECT COUNT(*) AS n FROM banned")
        self.assertEqual(row["n"], 1)

    def test_same_track_recognises_the_on_air_string(self):
        from airadio.runner import _same_track

        track = {"artist": "Kanye West", "title": "Runaway"}
        self.assertTrue(_same_track("Kanye West - Runaway", track))
        self.assertTrue(_same_track("kanye west - Runaway (Official Audio)", track))
        self.assertFalse(_same_track("Kanye West - Power", track))
        self.assertFalse(_same_track("", track))


class TestFallbackPlanner(RadioTestCase):
    def setUp(self):
        super().setUp()
        self.planner = Planner(self.cfg, self.db, self.library, LLM(self.cfg))

    def test_plans_without_an_llm(self):
        self.seed_library(artists=8, per_artist=4)
        plan = self.planner.plan_block(datetime(2026, 1, 1, 14, 0))
        self.assertEqual(plan["source"], "fallback")
        songs = [i for i in plan["items"] if i["kind"] == "song"]
        patter = [i for i in plan["items"] if i["kind"] == "patter"]
        self.assertGreater(len(songs), 2)
        self.assertGreater(len(patter), 0)

    def test_no_duplicate_songs_in_a_block(self):
        self.seed_library(artists=8, per_artist=4)
        plan = self.planner.plan_block(datetime(2026, 1, 1, 14, 0))
        ids = [i["track"]["id"] for i in plan["items"] if i["kind"] == "song"]
        self.assertEqual(len(ids), len(set(ids)))

    def test_show_note_names_the_actual_schedule_not_just_that_llm_is_down(self):
        # The site showed "fallback programming (no LLM)" with no hint of
        # what the fallback actually picked -- the mood text answers that.
        self.seed_library(artists=8, per_artist=4)
        plan = self.planner.plan_block(datetime(2026, 1, 1, 14, 0))
        mood_text = self.cfg.mood_for_hour(14)["mood"].strip()
        self.assertIn(mood_text, plan["show_note"])
        self.assertIn("no LLM", plan["show_note"])

    def test_artist_spacing_is_respected(self):
        self.seed_library(artists=8, per_artist=4)
        plan = self.planner.plan_block(datetime(2026, 1, 1, 14, 0))
        artists = [i["track"]["artist"] for i in plan["items"] if i["kind"] == "song"]
        spacing = int(self.cfg.get("planner.artist_spacing"))
        for index, artist in enumerate(artists):
            window = artists[max(0, index - spacing):index]
            self.assertNotIn(artist, window)

    def test_block_never_ends_on_patter(self):
        self.seed_library(artists=8, per_artist=4)
        plan = self.planner.plan_block(datetime(2026, 1, 1, 14, 0))
        self.assertEqual(plan["items"][-1]["kind"], "song")

    def test_empty_library_plans_nothing_and_does_not_crash(self):
        plan = self.planner.plan_block(datetime(2026, 1, 1, 14, 0))
        self.assertEqual(plan["items"], [])

    def test_validate_drops_hallucinated_ids(self):
        self.seed_library(artists=8, per_artist=4)
        candidates = self.planner.select_candidates([1, 5], 40)
        raw = ([{"type": "song", "id": t["id"]} for t in candidates[:6]]
               + [{"type": "song", "id": 999999},
                  {"type": "patter", "text": "A line."}])
        items = self.planner._validate(raw, candidates, 6)
        ids = [i["track"]["id"] for i in items if i["kind"] == "song"]
        self.assertEqual(len(ids), 6)
        self.assertNotIn(999999, ids)

    def test_validate_rejects_a_mostly_invented_plan(self):
        self.seed_library(artists=8, per_artist=4)
        candidates = self.planner.select_candidates([1, 5], 40)
        raw = [{"type": "song", "id": 900000 + n} for n in range(10)]
        self.assertEqual(self.planner._validate(raw, candidates, 10), [])

    def test_pool_is_spread_across_artists(self):
        # 3 artists with 20 tracks each: a random pool would hand the LLM a
        # lopsided menu and it would programme artists back to back.
        self.seed_library(artists=3, per_artist=20)
        pool = self.planner.select_candidates([1, 5], 30)
        counts: dict[str, int] = {}
        for track in pool:
            counts[track["artist"]] = counts.get(track["artist"], 0) + 1
        self.assertEqual(len(counts), 3)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1,
                             f"pool is lopsided: {counts}")

    def test_pool_respects_an_explicit_cap(self):
        self.seed_library(artists=4, per_artist=10)
        self.planner.cfg._data["planner"]["max_per_artist_in_pool"] = 2
        pool = self.planner.select_candidates([1, 5], 40)
        counts: dict[str, int] = {}
        for track in pool:
            counts[track["artist"]] = counts.get(track["artist"], 0) + 1
        self.assertTrue(all(n <= 2 for n in counts.values()), counts)

    def test_single_artist_library_still_returns_tracks(self):
        self.seed_library(artists=1, per_artist=8)
        self.assertEqual(len(self.planner.select_candidates([1, 5], 30)), 8)

    def test_artist_repeats_are_logged(self):
        self.seed_library(artists=2, per_artist=6)
        candidates = self.planner.select_candidates([1, 5], 20)
        same = [t for t in candidates if t["artist"] == candidates[0]["artist"]][:3]
        others = [t for t in candidates if t["artist"] != candidates[0]["artist"]][:3]
        # Interleave so there are enough songs to pass validation, but put two
        # by the same artist adjacent.
        raw = [{"type": "song", "id": same[0]["id"]},
               {"type": "song", "id": same[1]["id"]}]
        raw += [{"type": "song", "id": t["id"]} for t in others]
        with self.assertLogs("planner", level="WARNING") as captured:
            self.planner._validate(raw, candidates, 5)
        self.assertTrue(any("spacing window" in line for line in captured.output),
                        captured.output)

    def test_plans_for_when_the_block_airs_not_when_it_is_built(self):
        # The buffer is an hour deep, so a block built at 10:35 is heard at
        # 11:30. Planning against the build time picked the wrong time-of-day
        # slot and had the DJ announce a time that had already passed.
        self.seed_library(artists=8, per_artist=4)
        built_at = datetime(2026, 1, 1, 15, 30)
        plan = self.planner.plan_block(built_at, airs_in=3600)
        self.assertEqual(plan["mood_name"],
                         self.cfg.mood_for_hour(16)["name"])
        self.assertNotEqual(plan["mood_name"],
                            self.cfg.mood_for_hour(15)["name"])

    def test_no_buffer_means_plan_for_now(self):
        self.seed_library(artists=8, per_artist=4)
        plan = self.planner.plan_block(datetime(2026, 1, 1, 15, 30), airs_in=0)
        self.assertEqual(plan["mood_name"], self.cfg.mood_for_hour(15)["name"])

    def test_airtime_rolls_past_midnight(self):
        self.seed_library(artists=8, per_artist=4)
        plan = self.planner.plan_block(datetime(2026, 1, 1, 23, 30), airs_in=3600)
        self.assertEqual(plan["mood_name"], self.cfg.mood_for_hour(0)["name"])

    def test_clock_is_rounded_for_speech(self):
        from airadio.brain.planner import _round_clock

        self.assertEqual(_round_clock(datetime(2026, 1, 1, 11, 28)), "11:30")
        self.assertEqual(_round_clock(datetime(2026, 1, 1, 11, 2)), "11:00")
        # Rounding up past the hour must roll the hour, not produce 11:60.
        self.assertEqual(_round_clock(datetime(2026, 1, 1, 11, 58)), "12:00")
        self.assertEqual(_round_clock(datetime(2026, 1, 1, 23, 59)), "00:00")

    def named_artist_library(self) -> None:
        for artist in ("Westside Gunn", "Conway the Machine", "Boldy James"):
            for n in range(12):
                make_wav(self.cfg.path("library") / f"{artist} - Track {n}.wav",
                         seconds=3.0 + n * 0.2, tone=hash(artist) % 900 + n + 1)
        self.library.scan()

    def test_asking_for_an_artist_puts_them_in_the_pool(self):
        # The pool cap is roughly pool_size/artists, so without this an
        # explicit request reached the model as two tracks and an instruction
        # it could not act on.
        self.named_artist_library()
        self.planner.set_mood("im in the mood for some westside gunn")
        pool = self.planner.plan_block(datetime(2026, 1, 1, 14, 0))
        chosen = [i["track"]["artist"] for i in pool["items"]
                  if i["kind"] == "song"]
        self.assertGreaterEqual(chosen.count("Westside Gunn"), 4,
                                f"asked for Westside Gunn, got {chosen}")

    def test_focus_survives_the_diversity_cap(self):
        self.named_artist_library()
        focus = self.planner.select_candidates(
            [1, 5], 12, focus_artists=["Westside Gunn"])
        count = sum(1 for t in focus if t["artist"] == "Westside Gunn")
        self.assertGreaterEqual(count, 5, "the cap swallowed the focus tracks")

    def test_exemplars_are_present_but_do_not_dominate(self):
        # "think Smashing Pumpkins, Radiohead" illustrates a genre; it is not
        # an instruction to play only those two. Needs a realistic spread of
        # artists, since the baseline cap is pool_size / artist_count.
        self.seed_library(artists=20, per_artist=4)
        self.named_artist_library()
        plain = self.planner.select_candidates([1, 5], 40)
        with_exemplar = self.planner.select_candidates(
            [1, 5], 40, exemplar_artists=["Westside Gunn"])

        def count(pool):
            return sum(1 for t in pool if t["artist"] == "Westside Gunn")

        self.assertGreater(count(with_exemplar), count(plain),
                           "naming an artist changed nothing")
        self.assertLessEqual(count(with_exemplar), 8,
                             "exemplars took over the pool")

    def test_artists_named_in_finds_them_in_a_sentence(self):
        self.named_artist_library()
        found = self.planner.artists_named_in(
            "can we get some Westside Gunn tonight")
        self.assertEqual(found, ["Westside Gunn"])

    def test_artists_named_in_ignores_a_mood_with_no_names(self):
        self.named_artist_library()
        self.assertEqual(
            self.planner.artists_named_in("something darker and slower"), [])

    def test_mood_note_mentions_the_named_artist(self):
        from airadio.brain.planner import _mood_note

        note = _mood_note("some westside gunn", ["Westside Gunn"])
        self.assertIn("Westside Gunn", note)
        self.assertIn("spacing rule does not apply", note)
        self.assertEqual(_mood_note("", []), "")

    def test_mood_state_round_trip(self):
        self.planner.set_mood("darker and slower")
        self.assertEqual(self.planner.current_mood(), "darker and slower")
        self.planner.set_mood("")
        self.assertEqual(self.planner.current_mood(), "")


# -- queue ------------------------------------------------------------------


class TestQueue(RadioTestCase):
    def setUp(self):
        super().setUp()
        self.planner = Planner(self.cfg, self.db, self.library, LLM(self.cfg))
        self.seed_library(artists=8, per_artist=4)

    def build_raw(self) -> int:
        """Build a block, leaving patter lines pending as the brain does."""
        return self.queue.build_block(
            self.planner.plan_block(datetime(2026, 1, 1, 14, 0)))

    def build(self) -> int:
        """Build a block and drain the render queue.

        tts.engine is "none" in tests, so every patter line fails and is
        dropped -- leaving a songs-only queue, which is what the tests below
        care about.
        """
        count = self.build_raw()
        while self.queue.pending_count():
            self.queue.render_pending(limit=10)
        return count

    def test_build_block_queues_songs(self):
        queued = self.build_raw()
        self.assertGreater(queued, 0)
        self.assertEqual(queued,
                         self.queue.ready_count() + self.queue.pending_count())

    def test_patter_is_queued_pending_not_rendered_inline(self):
        # Building a block must be fast: rendering an hour of patter inline
        # blocks the brain, and chat requests queue behind it.
        self.build_raw()
        pending = self.db.query("SELECT * FROM queue_items WHERE status='pending'")
        self.assertGreater(len(pending), 0)
        self.assertTrue(all(row["kind"] in ("patter", "banter") for row in pending))
        self.assertTrue(all(row["path"] is None for row in pending))

    def test_failed_patter_is_dropped_not_left_blocking(self):
        # tts.engine is "none" in these tests, so every render fails.
        self.build_raw()
        while self.queue.pending_count():
            self.queue.render_pending(limit=5)
        self.assertEqual(
            self.db.one("SELECT COUNT(*) AS n FROM queue_items "
                        "WHERE status='pending'")["n"], 0)
        self.assertGreater(
            self.db.one("SELECT COUNT(*) AS n FROM queue_items "
                        "WHERE status='failed'")["n"], 0)

    def test_take_next_waits_for_an_unrendered_line(self):
        # The running order has to hold: playing the song that was meant to
        # follow a link, before the link, would sound broken.
        self.build_raw()
        first = self.db.one("SELECT id, kind FROM queue_items ORDER BY seq LIMIT 1")
        self.db.execute("UPDATE queue_items SET status='pending', path=NULL "
                        "WHERE id=?", (first["id"],))
        self.assertEqual(self.queue.take_next("ai", limit=3), [])

    def test_take_next_resumes_once_the_line_is_ready(self):
        self.build_raw()
        while self.queue.pending_count():
            self.queue.render_pending(limit=5)
        self.assertEqual(len(self.queue.take_next("ai", limit=3)), 3)

    def test_take_next_claims_items_once(self):
        self.build()
        first = self.queue.take_next("ai", limit=2)
        second = self.queue.take_next("ai", limit=2)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertFalse({i["id"] for i in first} & {i["id"] for i in second})

    def test_take_next_does_not_mark_tracks_played(self):
        # Handing a track to liquidsoap is not the same as it having aired --
        # liquidsoap can hold several items in its own buffer ahead of
        # whatever is actually audible. Stamping "played" here (the old
        # behaviour) made a track look already played while it was still
        # sitting in the upcoming queue.
        self.build()
        item = self.queue.take_next("ai", limit=1)[0]
        row = self.db.one("SELECT play_count, last_played_at FROM tracks WHERE id=?",
                          (item["track_id"],))
        self.assertEqual(row["play_count"], 0)
        self.assertIsNone(row["last_played_at"])

    def test_take_next_drops_vanished_files(self):
        self.build()
        row = self.db.one("SELECT id, path FROM queue_items WHERE status='ready' "
                          "ORDER BY seq LIMIT 1")
        Path(row["path"]).unlink()
        claimed = self.queue.take_next("ai", limit=1)
        self.assertNotEqual(claimed[0]["id"] if claimed else None, row["id"])
        after = self.db.one("SELECT status FROM queue_items WHERE id=?", (row["id"],))
        self.assertEqual(after["status"], "failed")

    def test_requests_are_a_separate_tier(self):
        self.build()
        track = dict(self.db.one("SELECT * FROM tracks LIMIT 1"))
        self.queue.enqueue_request(track)
        self.assertEqual(len(self.queue.take_next("request", limit=5)), 1)

    def test_needs_block_flips_once_the_buffer_is_full(self):
        self.assertTrue(self.queue.needs_block())
        self.db.execute(
            "INSERT INTO queue_items(seq, kind, tier, path, duration, status, "
            "created_at) VALUES(1,'song','ai','x.mp3',99999,'ready',0)")
        self.assertFalse(self.queue.needs_block())

    def test_pushed_items_still_count_towards_the_buffer(self):
        self.build()
        before = self.queue.ready_seconds()
        self.queue.take_next("ai", limit=3)
        self.assertAlmostEqual(self.queue.ready_seconds(), before, places=3,
                               msg="handing an item to liquidsoap must not "
                                   "change how much audio is waiting")

    def test_reconcile_retires_items_liquidsoap_has_played(self):
        self.build()
        self.queue.take_next("ai", limit=5)
        before = self.queue.ready_seconds()
        # Liquidsoap reports 2 still queued: the other 3 have been played.
        self.queue.reconcile_pushed("ai", 2)
        self.assertLess(self.queue.ready_seconds(), before)
        done = self.db.one("SELECT COUNT(*) AS n FROM queue_items WHERE status='done'")
        self.assertEqual(done["n"], 3)

    def test_buffer_does_not_grow_without_bound(self):
        # The bug this guards: pushed items were never retired, so the buffer
        # reading climbed forever and the planner stopped building blocks.
        self.build()
        for _ in range(6):
            self.queue.take_next("ai", limit=2)
            self.queue.reconcile_pushed("ai", 1)
        remaining = self.db.one(
            "SELECT COUNT(*) AS n FROM queue_items WHERE status IN ('ready','pushed')")
        self.assertLessEqual(
            self.queue.ready_seconds(),
            remaining["n"] * 600,
            "buffer is counting audio that has already been played")

    def test_reconcile_marks_only_the_tracks_that_actually_finished(self):
        self.build()
        items = self.queue.take_next("ai", limit=3)
        # Liquidsoap reports 1 still queued: the other 2 have actually aired.
        self.queue.reconcile_pushed("ai", 1)

        for item in items[:2]:
            row = self.db.one(
                "SELECT play_count, last_played_at FROM tracks WHERE id=?",
                (item["track_id"],))
            self.assertEqual(row["play_count"], 1)
            self.assertIsNotNone(row["last_played_at"])

        row = self.db.one("SELECT play_count, last_played_at FROM tracks WHERE id=?",
                          (items[2]["track_id"],))
        self.assertEqual(row["play_count"], 0)
        self.assertIsNone(row["last_played_at"])

    def test_reconcile_keeps_the_newest_items(self):
        self.build()
        pushed = self.queue.take_next("ai", limit=4)
        self.queue.reconcile_pushed("ai", 2)
        still = self.db.query(
            "SELECT id FROM queue_items WHERE status='pushed' ORDER BY seq")
        self.assertEqual([row["id"] for row in still],
                         [item["id"] for item in pushed[-2:]])

    def test_reconcile_does_not_touch_the_other_tier(self):
        self.build()
        track = dict(self.db.one("SELECT * FROM tracks LIMIT 1"))
        self.queue.enqueue_request(track)
        self.queue.take_next("request", limit=1)
        self.queue.take_next("ai", limit=3)
        self.queue.reconcile_pushed("ai", 0)
        row = self.db.one(
            "SELECT status FROM queue_items WHERE tier='request' ORDER BY id DESC LIMIT 1")
        self.assertEqual(row["status"], "pushed")

    def test_clear_pending_only_drops_unplayed(self):
        self.build()
        self.queue.take_next("ai", limit=2)
        before = self.db.one("SELECT COUNT(*) AS n FROM queue_items")["n"]
        self.queue.clear_pending("ai")
        after = self.db.one(
            "SELECT COUNT(*) AS n FROM queue_items WHERE status='pushed'")["n"]
        self.assertEqual(after, 2)
        self.assertLess(after, before)


# -- weather ----------------------------------------------------------------


class FakeWeather:
    """Stands in for Open-Meteo. No test is allowed to touch the network."""

    def __init__(self, reading: dict | None = None):
        self.enabled = reading is not None
        self._reading = reading

    def current(self, force: bool = False) -> dict | None:
        return self._reading

    def briefing(self, include_outlook: bool = True) -> str:
        if not self._reading:
            return ""
        line = (f"Weather in {self._reading['place']} right now: "
                f"{self._reading['temp_c']} degrees Celsius.")
        if include_outlook:
            line += f" Later a high of {self._reading['high_c']}."
        return line


SAMPLE_READING = {
    "place": "Eindhoven", "temp_c": 11, "feels_c": 9, "condition": "light rain",
    "wind_kmh": 30, "is_day": True, "high_c": 14, "low_c": 7,
    "today": "showers", "rain_chance": 70, "fetched_at": 0.0,
}


class TestWeather(unittest.TestCase):
    def test_disabled_weather_reports_nothing(self):
        cfg = Config({"weather": {"enabled": False}}, Path("."))
        self.assertIsNone(Weather(cfg).current())
        self.assertEqual(Weather(cfg).briefing(), "")

    def test_briefing_is_built_from_a_reading(self):
        cfg = Config({"weather": {"enabled": True}}, Path("."))
        weather = Weather(cfg)
        weather._cache = dict(SAMPLE_READING)
        weather._cached_at = time.time()
        brief = weather.briefing()
        self.assertIn("Eindhoven", brief)
        self.assertIn("11 degrees", brief)
        self.assertIn("light rain", brief)
        # Ten degrees of wind chill is worth saying; two degrees is not.
        self.assertNotIn("feels like", brief)
        self.assertIn("70 percent", brief)

    def test_briefing_can_leave_out_the_day_ahead(self):
        cfg = Config({"weather": {"enabled": True}}, Path("."))
        weather = Weather(cfg)
        weather._cache = dict(SAMPLE_READING)
        weather._cached_at = time.time()
        self.assertNotIn("high of", weather.briefing(include_outlook=False))

    def test_a_failed_fetch_keeps_serving_the_last_reading(self):
        cfg = Config({"weather": {"enabled": True}}, Path("."))
        weather = Weather(cfg)
        weather._cache = dict(SAMPLE_READING)
        weather._cached_at = 0.0          # stale, so a refresh is due
        weather._fetch = lambda: None     # and the refresh fails
        self.assertEqual(weather.current()["temp_c"], 11)
        self.assertGreater(weather._quiet_until, time.time())

    def test_spoken_numbers_are_words(self):
        self.assertEqual(_spoken_number(11), "eleven")
        self.assertEqual(_spoken_number(21), "twenty one")
        self.assertEqual(_spoken_number(-3), "minus three")
        self.assertEqual(_spoken_number(100), "a hundred")
        self.assertEqual(_spoken_number(None), "")

    def test_template_weather_line_has_no_digits(self):
        line = _template_weather(SAMPLE_READING, outlook=True)
        self.assertNotRegex(line, r"[0-9]")
        self.assertIn("Eindhoven", line)
        self.assertIn("eleven degrees", line)

    def test_template_weather_survives_a_missing_reading(self):
        self.assertEqual(_template_weather(None), "")
        self.assertEqual(_template_weather({"place": "x"}), "")


# -- two hosts --------------------------------------------------------------


class TestHostsAndSegments(RadioTestCase):
    def setUp(self):
        super().setUp()
        self.planner = Planner(self.cfg, self.db, self.library, LLM(self.cfg),
                               FakeWeather(dict(SAMPLE_READING)))

    def _segments(self, **overrides):
        """Rewrite dj.segments in place for a deterministic roll."""
        self.cfg._data["dj"] = dict(self.cfg._data["dj"])
        self.cfg._data["dj"]["segments"] = {
            "banter_per_block": [0, 0], "banter_turns": [4, 4],
            "music_notes_per_block": [0, 0], "weather_chance": 0.0,
            "weather_outlook_hours": [5, 11], **overrides}

    def test_hosts_come_from_config(self):
        self.assertEqual(self.planner.host_names(), ["Ray", "Nina"])

    def test_a_host_with_no_voice_is_left_out_of_the_show(self):
        # The whole point: a line written for Nina but rendered in Ray's voice
        # would sound like one person having an argument with themselves.
        self.assertEqual(self.planner.host_names(["Ray"]), ["Ray"])

    def test_host_names_never_comes_back_empty(self):
        self.assertEqual(self.planner.host_names(["Nobody"]), ["Ray"])

    def test_quiet_hour_asks_for_nothing_extra(self):
        self._segments()
        brief = self.planner.segment_brief(datetime(2026, 1, 1, 14, 0),
                                           ["Ray", "Nina"], True)
        self.assertEqual(brief, {"banter": 0, "banter_turns": 4,
                                 "music_notes": 0, "weather": ""})
        self.assertIn("quiet one",
                      self.planner._segment_plan_text(brief, ["Ray", "Nina"]))

    def test_a_single_host_never_gets_a_conversation(self):
        self._segments(banter_per_block=[2, 2])
        brief = self.planner.segment_brief(datetime(2026, 1, 1, 14, 0),
                                           ["Ray"], True)
        self.assertEqual(brief["banter"], 0)

    def test_weather_slot_follows_the_clock(self):
        self._segments(weather_chance=1.0)
        morning = self.planner.segment_brief(datetime(2026, 1, 1, 8, 0),
                                             ["Ray", "Nina"], True)
        evening = self.planner.segment_brief(datetime(2026, 1, 1, 22, 0),
                                             ["Ray", "Nina"], True)
        self.assertEqual(morning["weather"], "outlook")
        self.assertEqual(evening["weather"], "now")

    def test_no_weather_slot_when_there_is_no_weather(self):
        self._segments(weather_chance=1.0)
        brief = self.planner.segment_brief(datetime(2026, 1, 1, 8, 0),
                                           ["Ray", "Nina"], False)
        self.assertEqual(brief["weather"], "")

    def test_weather_text_is_only_offered_when_a_slot_asked_for_it(self):
        self.assertEqual(self.planner._weather_text({"weather": ""}), "")
        self.assertIn("high of", self.planner._weather_text({"weather": "outlook"}))
        self.assertNotIn("high of", self.planner._weather_text({"weather": "now"}))


class TestGenreFiltering(RadioTestCase):
    """The reported bug: energy alone cannot tell a rock hour from a jazz
    one at the same tempo, so the candidate pool needs to actually respect
    a mood_map entry's genres, not just its energy band."""

    def setUp(self):
        super().setUp()
        self.planner = Planner(self.cfg, self.db, self.library, LLM(self.cfg))

    def _add_track(self, artist: str, title: str, genre: str = "",
                   tags: str = "", energy: int = 3) -> None:
        path = self.cfg.path("library") / f"{normalize(artist)}-{normalize(title)}.mp3"
        make_wav(path, seconds=180.0)
        self.db.execute(
            "INSERT INTO tracks(path, dedupe_key, title, artist, genre, tags, "
            "energy, duration, added_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (str(path), dedupe_key(artist, title), title, artist, genre, tags,
             energy, 180.0, 0))

    def _seed_mixed_library(self, per_genre: int = 20) -> None:
        for n in range(per_genre):
            self._add_track("Rock Artist", f"Rock Track {n}",
                            genre="alternative rock", tags="rock, 90s, alternative")
            self._add_track("Jazz Artist", f"Jazz Track {n}",
                            genre="jazz", tags="jazz, smooth, lounge")

    def test_query_by_genre_matches_loosely(self):
        self._seed_mixed_library()
        rows = self.planner._query_by_genre(["rock"], 0, 1, 5, 60)
        artists = {row["artist"] for row in rows}
        self.assertEqual(artists, {"Rock Artist"})

    def test_query_by_genre_matches_several_genres_at_once(self):
        self._seed_mixed_library()
        self._add_track("Rap Artist", "Rap Track", genre="hip hop",
                        tags="hip hop, rap")
        rows = self.planner._query_by_genre(["rock", "hip hop"], 0, 1, 5, 60)
        artists = {row["artist"] for row in rows}
        self.assertEqual(artists, {"Rock Artist", "Rap Artist"})

    def test_select_candidates_excludes_the_wrong_genre(self):
        self._seed_mixed_library()
        pool = self.planner.select_candidates([1, 5], 60, genres=["rock"])
        artists = {track["artist"] for track in pool}
        self.assertEqual(artists, {"Rock Artist"},
                         "jazz should never have reached the pool for a rock hour")

    def test_select_candidates_widens_past_genre_when_too_few_match(self):
        # Only two rock tracks exist -- nowhere near enough for a real hour.
        # An hour that ignores genre beats one with two songs in it.
        self._add_track("Rock Artist", "Track 1", genre="rock", tags="rock")
        self._add_track("Rock Artist", "Track 2", genre="rock", tags="rock")
        for n in range(20):
            self._add_track("Jazz Artist", f"Jazz {n}", genre="jazz", tags="jazz")
        pool = self.planner.select_candidates([1, 5], 60, genres=["rock"])
        artists = {track["artist"] for track in pool}
        self.assertIn("Jazz Artist", artists,
                      "the genre filter should have been dropped for this "
                      "thin a match, not starved the pool")

    def test_no_genres_means_unfiltered_like_before(self):
        self._seed_mixed_library()
        pool = self.planner.select_candidates([1, 5], 60, genres=None)
        artists = {track["artist"] for track in pool}
        self.assertEqual(artists, {"Rock Artist", "Jazz Artist"})

    def test_plan_block_passes_the_profile_genres_through(self):
        self._seed_mixed_library()
        self.cfg._data["planner"] = dict(self.cfg._data["planner"])
        self.cfg._data["planner"]["mood_map"] = [
            {"hours": [0, 24], "name": "rock hour",
             "mood": "rock music", "energy": [1, 5], "genres": ["rock"]},
        ]
        plan = self.planner.plan_block(datetime(2026, 1, 1, 14, 0))
        artists = {item["track"]["artist"] for item in plan["items"]
                  if item["kind"] == "song"}
        self.assertEqual(artists, {"Rock Artist"})

    def test_plan_block_recovers_genre_filtering_after_a_local_override_drops_it(self):
        # The actual reported bug, end to end: a config.local.yaml that
        # replaces planner.mood_map with entries carrying no genres: (because
        # it predates that field, or only customised the mood text) used to
        # mean every hour was an unfiltered shuffle of the whole library --
        # exactly the rap/rock/jazz-all-in-one-block symptom reported. It
        # must self-heal via planner.default_genres as long as the slot name
        # ("evening") still matches the shipped one.
        self._seed_mixed_library()
        self.cfg._data["planner"] = dict(self.cfg._data["planner"])
        self.cfg._data["planner"]["mood_map"] = [
            {"hours": [0, 24], "name": "evening",
             "mood": "my own jazzy words", "energy": [1, 5]},
        ]
        plan = self.planner.plan_block(datetime(2026, 1, 1, 18, 0))
        artists = {item["track"]["artist"] for item in plan["items"]
                  if item["kind"] == "song"}
        self.assertEqual(artists, {"Jazz Artist"})


class TestBanterValidation(RadioTestCase):
    def setUp(self):
        super().setUp()
        self.planner = Planner(self.cfg, self.db, self.library, LLM(self.cfg))
        self.seed_library(artists=8, per_artist=4)
        self.candidates = self.planner.select_candidates([1, 5], 40)

    def _validate(self, extra, names=("Ray", "Nina")):
        raw = ([{"type": "song", "id": t["id"]} for t in self.candidates[:6]]
               + extra
               + [{"type": "song", "id": self.candidates[6]["id"]}])
        return self.planner._validate(raw, self.candidates, 6, list(names))

    def test_a_conversation_survives_validation(self):
        items = self._validate([{"type": "banter", "lines": [
            {"host": "Ray", "text": "That was good."},
            {"host": "Nina", "text": "It was. What is next?"},
            {"host": "Ray", "text": "Something slower."}]}])
        banter = [i for i in items if i["kind"] == "banter"]
        self.assertEqual(len(banter), 1)
        self.assertEqual([turn["host"] for turn in banter[0]["lines"]],
                         ["Ray", "Nina", "Ray"])

    def test_two_turns_from_one_host_are_merged_not_dropped(self):
        turns = _clean_exchange([
            {"host": "Ray", "text": "First thought."},
            {"host": "Ray", "text": "Second thought."},
            {"host": "Nina", "text": "My turn."}], ["Ray", "Nina"], 90)
        self.assertEqual([turn["host"] for turn in turns], ["Ray", "Nina"])
        self.assertIn("First thought", turns[0]["text"])
        self.assertIn("Second thought", turns[0]["text"])

    def test_an_invented_host_is_mapped_onto_a_real_one(self):
        turns = _clean_exchange([
            {"host": "Dave", "text": "Hello."},
            {"host": "Susan", "text": "Hello back."}], ["Ray", "Nina"], 90)
        self.assertEqual([turn["host"] for turn in turns], ["Ray", "Nina"])

    def test_a_conversation_collapses_to_one_voice_when_alone(self):
        turns = _clean_exchange([
            {"host": "Ray", "text": "One."},
            {"host": "Nina", "text": "Two."}], ["Ray"], 90)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["host"], "Ray")
        self.assertIn("One", turns[0]["text"])
        self.assertIn("Two", turns[0]["text"])

    def test_a_one_line_conversation_becomes_a_plain_line(self):
        items = self._validate([{"type": "banter", "lines": [
            {"host": "Ray", "text": "Only me tonight."}]}])
        self.assertNotIn("banter", [i["kind"] for i in items])
        self.assertIn("Only me tonight.",
                      [i.get("text") for i in items if i["kind"] == "patter"])

    def test_malformed_lines_do_not_crash_the_plan(self):
        items = self._validate([{"type": "banter", "lines": "not a list"},
                                {"type": "banter"},
                                {"type": "banter", "lines": [None, 7]}])
        self.assertGreater(len([i for i in items if i["kind"] == "song"]), 2)

    def test_patter_keeps_the_host_it_was_written_for(self):
        items = self._validate([{"type": "patter", "host": "Nina",
                                 "text": "Just me for a second."}])
        line = [i for i in items if i["kind"] == "patter"][0]
        self.assertEqual(line["host"], "Nina")

    def test_a_block_never_ends_on_a_conversation(self):
        raw = ([{"type": "song", "id": t["id"]} for t in self.candidates[:6]]
               + [{"type": "banter", "lines": [
                   {"host": "Ray", "text": "Bye."},
                   {"host": "Nina", "text": "Night."}]}])
        items = self.planner._validate(raw, self.candidates, 6, ["Ray", "Nina"])
        self.assertEqual(items[-1]["kind"], "song")

    def test_stacked_talk_items_collapse_to_one(self):
        # The reported bug: the model wrote an opening link, another link, a
        # conversation, then five more links before ever reaching a song --
        # the site's "up next" showed eight straight talk items and no music
        # at all.
        items = self._validate([
            {"type": "patter", "host": "Nina", "text": "Evening."},
            {"type": "patter", "host": "Nina", "text": "Settling in."},
            {"type": "banter", "lines": [
                {"host": "Ray", "text": "Good one."},
                {"host": "Nina", "text": "Always is."}]},
            {"type": "patter", "host": "Ray", "text": "One."},
            {"type": "patter", "host": "Nina", "text": "Two."},
            {"type": "patter", "host": "Ray", "text": "Three."},
            {"type": "patter", "host": "Nina", "text": "Four."},
            {"type": "patter", "host": "Ray", "text": "Five."},
        ])
        kinds = [item["kind"] for item in items]
        # Only the first of that whole run should have survived; everything
        # else in the pile-up gets dropped, not the songs around it.
        talk_run = kinds[6:-1]  # between the six opening songs and the last
        self.assertEqual(len(talk_run), 1, kinds)
        self.assertEqual(kinds.count("song"), 7)

    def test_drop_consecutive_talk_keeps_a_normal_alternating_plan(self):
        items = [
            {"kind": "patter", "text": "a"},
            {"kind": "song", "track": {}},
            {"kind": "banter", "lines": []},
            {"kind": "song", "track": {}},
            {"kind": "patter", "text": "b"},
            {"kind": "song", "track": {}},
        ]
        self.assertEqual(_drop_consecutive_talk(items), items)

    def test_drop_consecutive_talk_keeps_the_first_of_a_run(self):
        items = [
            {"kind": "song", "track": {}},
            {"kind": "patter", "text": "first"},
            {"kind": "banter", "lines": []},
            {"kind": "patter", "text": "third"},
            {"kind": "song", "track": {}},
        ]
        fixed = _drop_consecutive_talk(items)
        self.assertEqual([item["kind"] for item in fixed],
                         ["song", "patter", "song"])
        self.assertEqual(fixed[1]["text"], "first")

    def test_drop_consecutive_talk_handles_an_all_talk_list(self):
        items = [{"kind": "patter", "text": "a"}, {"kind": "banter", "lines": []},
                {"kind": "patter", "text": "c"}]
        self.assertEqual(_drop_consecutive_talk(items),
                         [{"kind": "patter", "text": "a"}])


class TestBanterQueueing(RadioTestCase):
    def setUp(self):
        super().setUp()
        self.seed_library(artists=4, per_artist=3)
        self.track = dict(self.db.query("SELECT * FROM tracks LIMIT 1")[0])

    def _plan(self):
        return {"items": [
            {"kind": "patter", "host": "Ray", "text": "Evening."},
            {"kind": "banter", "lines": [{"host": "Ray", "text": "One."},
                                         {"host": "Nina", "text": "Two."}]},
            {"kind": "song", "track": self.track},
        ], "source": "test", "mood_name": "night", "show_note": ""}

    def test_a_conversation_is_stored_as_a_single_row(self):
        self.queue.build_block(self._plan())
        rows = self.db.query("SELECT * FROM queue_items WHERE kind='banter'")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["host"], "Ray & Nina")
        turns = json.loads(rows[0]["text"])
        self.assertEqual([turn["host"] for turn in turns], ["Ray", "Nina"])

    def test_a_plain_line_remembers_its_host(self):
        self.queue.build_block(self._plan())
        row = self.db.one("SELECT * FROM queue_items WHERE kind='patter'")
        self.assertEqual(row["host"], "Ray")

    def test_rendering_routes_a_conversation_to_the_exchange_renderer(self):
        self.queue.build_block(self._plan())
        seen = {}

        def fake_exchange(lines, name_hint="banter"):
            seen["lines"] = lines
            return None            # a failed render is dropped, which is fine

        self.queue.tts.render_exchange = fake_exchange
        self.queue.tts.render = lambda *a, **k: None
        while self.queue.pending_count():
            self.queue.render_pending(limit=5)
        self.assertEqual([turn["host"] for turn in seen["lines"]],
                         ["Ray", "Nina"])

    def test_a_corrupt_conversation_is_dropped_not_fatal(self):
        self.queue.build_block(self._plan())
        self.db.execute("UPDATE queue_items SET text='{not json' "
                        "WHERE kind='banter'")
        while self.queue.pending_count():
            self.queue.render_pending(limit=5)
        self.assertEqual(self.queue.pending_count(), 0)


class TestVoiceSelection(RadioTestCase):
    def test_the_first_host_inherits_the_station_voice(self):
        data = dict(BASE_CONFIG)
        data["tts"] = dict(data["tts"], engine="piper_cli",
                           voice_model="/voices/en_US-ryan-high.onnx",
                           length_scale=1.4)
        cfg = Config(data, self.tmp)
        tts = TTS(cfg)
        self.assertEqual(tts.voices["Ray"].model,
                         "/voices/en_US-ryan-high.onnx")
        self.assertEqual(tts.voices["Ray"].length_scale, 1.4)

    def test_the_co_host_voice_is_looked_for_beside_the_first(self):
        data = dict(BASE_CONFIG)
        data["tts"] = dict(data["tts"], engine="piper_cli",
                           voice_model="/voices/en_US-ryan-high.onnx")
        tts = TTS(Config(data, self.tmp))
        self.assertTrue(tts.voices["Nina"].model.endswith(
            "en_GB-jenny_dioco-medium.onnx"))
        self.assertIn("/voices", tts.voices["Nina"].model.replace("\\", "/"))

    def test_a_blank_length_scale_falls_back_to_the_station_setting(self):
        data = dict(BASE_CONFIG)
        data["tts"] = dict(data["tts"], length_scale=1.25)
        tts = TTS(Config(data, self.tmp))
        self.assertEqual(tts.voices["Nina"].length_scale, 1.25)

    def test_only_hosts_with_a_model_on_disk_can_speak(self):
        voice = self.cfg.path("state") / "en_US-ryan-high.onnx"
        voice.write_bytes(b"not really a model, but it exists")
        data = dict(BASE_CONFIG)
        data["tts"] = dict(data["tts"], engine="piper_cli",
                           voice_model=str(voice))
        tts = TTS(Config(data, self.tmp))
        self.assertEqual(tts.hosts, ["Ray"])
        ok, detail = tts.check_exchange()
        self.assertFalse(ok)
        self.assertIn("one voice", detail)

    def test_disabled_tts_has_no_speaking_hosts(self):
        self.assertEqual(TTS(self.cfg).hosts, [])

    def test_hosts_can_set_their_own_expressiveness(self):
        # The shipped config gives Nina more lift than Ray on purpose.
        tts = TTS(self.cfg)
        self.assertGreater(tts.voices["Nina"].noise_scale,
                           tts.voices["Ray"].noise_scale)
        self.assertGreater(tts.voices["Nina"].noise_w,
                           tts.voices["Ray"].noise_w)

    def test_a_blank_noise_value_falls_back_to_the_station_setting(self):
        data = dict(BASE_CONFIG)
        data["tts"] = dict(data["tts"], noise_scale=0.77, noise_w=0.91)
        data["dj"] = dict(data["dj"])
        data["dj"]["hosts"] = [
            {"name": "Solo", "voice_model": "", "length_scale": None,
             "noise_scale": None, "noise_w": None}]
        tts = TTS(Config(data, self.tmp))
        self.assertEqual(tts.voices["Solo"].noise_scale, 0.77)
        self.assertEqual(tts.voices["Solo"].noise_w, 0.91)


class TestExpressivenessFlags(RadioTestCase):
    """noise_scale/noise_w (Piper's expressiveness knobs) actually reach the
    command line, and a rejection falls back to plain audio rather than no
    audio at all."""

    def _tts_with(self, noise_scale, noise_w) -> TTS:
        data = dict(BASE_CONFIG)
        data["tts"] = dict(data["tts"], engine="piper_python")
        data["dj"] = dict(data["dj"])
        data["dj"]["hosts"] = [{
            "name": "Solo", "voice_model": "/voices/test.onnx",
            "length_scale": 1.0, "noise_scale": noise_scale,
            "noise_w": noise_w,
        }]
        return TTS(Config(data, self.tmp))

    def _run_with_fake_subprocess(self, tts: TTS, fake_run) -> Path:
        import airadio.tts as tts_module
        out = self.tmp / "out.wav"
        # _run only checks that the file exists and is not tiny; it never
        # looks at the content, so a fake is enough to stand in for a real
        # subprocess actually writing audio.
        out.write_bytes(b"x" * 2000)
        original = tts_module.subprocess.run
        tts_module.subprocess.run = fake_run
        try:
            return tts._synthesize("test line", out, tts.voices["Solo"])
        finally:
            tts_module.subprocess.run = original

    def test_non_default_noise_values_reach_the_command_line(self):
        tts = self._tts_with(0.9, 1.0)
        captured = {}

        def fake_run(command, **kwargs):
            captured["cmd"] = command
            return _FakeCompleted(0)

        self._run_with_fake_subprocess(tts, fake_run)
        cmd = captured["cmd"]
        self.assertIn("0.9", cmd)
        self.assertIn("1.0", cmd)
        self.assertTrue(any("noise_scale" in part or "noise-scale" in part
                            for part in cmd), cmd)
        self.assertTrue(any("noise_w" in part or "noise-w" in part
                            for part in cmd), cmd)

    def test_stock_noise_values_add_no_flags_at_all(self):
        # Exactly Piper's own defaults -- nothing new to ask for.
        tts = self._tts_with(0.667, 0.8)
        captured = {}

        def fake_run(command, **kwargs):
            captured["cmd"] = command
            return _FakeCompleted(0)

        self._run_with_fake_subprocess(tts, fake_run)
        self.assertFalse(any("noise" in part for part in captured["cmd"]))

    def test_a_rejected_noise_flag_falls_back_to_plain_audio(self):
        tts = self._tts_with(0.9, 1.0)
        attempts = []

        def fake_run(command, **kwargs):
            attempts.append(command)
            rejected = any("noise" in part for part in command)
            return _FakeCompleted(1 if rejected else 0)

        result = self._run_with_fake_subprocess(tts, fake_run)
        self.assertIsNotNone(result, "a rejected flag must not silence the line")
        self.assertEqual(len(attempts), 2, "one failed attempt, one clean retry")
        self.assertFalse(tts._noise_supported,
                         "future lines should stop trying the flag it rejected")


class _FakeCompleted:
    def __init__(self, returncode: int, stderr: bytes = b""):
        self.returncode = returncode
        self.stderr = stderr


def _output_path(command):
    """Where a faked command would have written its audio, going by which
    output flag is present -- edge-tts, piper, or a bare ffmpeg convert."""
    for flag in ("--write-media", "-f", "--output_file", "--output-file"):
        if flag in command:
            return Path(command[command.index(flag) + 1])
    if "-i" in command:
        return Path(command[-1])
    return None


class TestEdgeTTSFallback(RadioTestCase):
    """engine: edge_tts tries Microsoft's free neural voices first and falls
    back to piper per line whenever edge-tts is unreachable -- offline,
    rate-limited, blocked, or simply not installed. Unofficial endpoint, so
    the fallback is what keeps patter alive rather than going silent."""

    def _tts_with(self, edge_voice="en-US-AriaNeural", fallback_engine="piper_cli",
                 edge_rate=None, edge_pitch=None):
        data = dict(BASE_CONFIG)
        data["tts"] = dict(data["tts"], engine="edge_tts",
                           fallback_engine=fallback_engine,
                           voice_model="/voices/test.onnx")
        data["dj"] = dict(data["dj"])
        host = {"name": "Solo", "voice_model": "/voices/test.onnx",
               "edge_voice": edge_voice, "length_scale": 1.0}
        if edge_rate is not None:
            host["edge_rate"] = edge_rate
        if edge_pitch is not None:
            host["edge_pitch"] = edge_pitch
        data["dj"]["hosts"] = [host]
        return TTS(Config(data, self.tmp))

    def _patched(self, which_map, run_fn):
        import airadio.tts as tts_module
        originals = (tts_module.shutil.which, tts_module.subprocess.run)
        tts_module.shutil.which = lambda name: which_map.get(name)
        tts_module.subprocess.run = run_fn
        return tts_module, originals

    def _restore(self, tts_module, originals):
        tts_module.shutil.which, tts_module.subprocess.run = originals

    def test_edge_tts_is_used_when_it_works(self):
        tts = self._tts_with()
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            out = _output_path(command)
            if out is not None:
                out.write_bytes(b"x" * 2000)
            return _FakeCompleted(0)

        tts_module, originals = self._patched(
            {"edge-tts": "/usr/bin/edge-tts", "ffmpeg": "/usr/bin/ffmpeg"},
            fake_run)
        try:
            result = tts._synthesize("hello", self.tmp / "out.wav",
                                     tts.voices["Solo"])
        finally:
            self._restore(tts_module, originals)

        self.assertIsNotNone(result)
        self.assertEqual(len(calls), 2, "one edge-tts call, one ffmpeg convert")
        self.assertIn("edge-tts", calls[0][0])
        self.assertIn("en-US-AriaNeural", calls[0])
        self.assertTrue(all("-m" not in c for c in calls), "piper never ran")

    def test_a_failed_edge_request_falls_back_to_piper(self):
        tts = self._tts_with()
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if "edge-tts" in command[0]:
                return _FakeCompleted(1, b"blocked")
            out = _output_path(command)
            if out is not None:
                out.write_bytes(b"x" * 2000)
            return _FakeCompleted(0)

        tts_module, originals = self._patched(
            {"edge-tts": "/usr/bin/edge-tts", "ffmpeg": "/usr/bin/ffmpeg",
             "piper": "/usr/bin/piper"}, fake_run)
        try:
            result = tts._synthesize("hello", self.tmp / "out.wav",
                                     tts.voices["Solo"])
        finally:
            self._restore(tts_module, originals)

        self.assertIsNotNone(result, "a blocked edge-tts must not silence the line")
        self.assertEqual(len(calls), 2, "one failed edge-tts call, one piper retry")
        self.assertIn("edge-tts", calls[0][0])
        self.assertIn("-m", calls[1])

    def test_missing_ffmpeg_skips_edge_and_goes_straight_to_piper(self):
        # edge-tts only speaks mp3, so with no ffmpeg to convert it there is
        # nothing useful edge-tts could do -- do not even call it.
        tts = self._tts_with()
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            out = _output_path(command)
            if out is not None:
                out.write_bytes(b"x" * 2000)
            return _FakeCompleted(0)

        tts_module, originals = self._patched(
            {"edge-tts": "/usr/bin/edge-tts", "piper": "/usr/bin/piper"},
            fake_run)
        try:
            result = tts._synthesize("hello", self.tmp / "out.wav",
                                     tts.voices["Solo"])
        finally:
            self._restore(tts_module, originals)

        self.assertIsNotNone(result)
        self.assertEqual(len(calls), 1, "no wasted edge-tts call")
        self.assertIn("-m", calls[0])

    def test_check_reports_the_fallback_by_name_when_edge_is_down(self):
        tts = self._tts_with()

        def fake_run(command, **kwargs):
            if "edge-tts" in command[0]:
                return _FakeCompleted(1, b"blocked")
            out = _output_path(command)
            if out is not None:
                out.write_bytes(b"x" * 2000)
            return _FakeCompleted(0)

        tts_module, originals = self._patched(
            {"edge-tts": "/usr/bin/edge-tts", "ffmpeg": "/usr/bin/ffmpeg",
             "piper": "/usr/bin/piper"}, fake_run)
        try:
            ok, detail = tts.check()
        finally:
            self._restore(tts_module, originals)

        self.assertTrue(ok, detail)
        self.assertIn("fallback", detail)
        self.assertIn("piper_cli", detail)

    def test_check_fails_when_both_edge_and_the_fallback_are_down(self):
        tts = self._tts_with()

        def fake_run(command, **kwargs):
            return _FakeCompleted(1, b"nope")

        tts_module, originals = self._patched(
            {"edge-tts": "/usr/bin/edge-tts", "ffmpeg": "/usr/bin/ffmpeg",
             "piper": "/usr/bin/piper"}, fake_run)
        try:
            ok, detail = tts.check()
        finally:
            self._restore(tts_module, originals)

        self.assertFalse(ok)
        self.assertIn("piper_cli", detail)

    def test_a_host_with_only_an_edge_voice_can_still_speak(self):
        # No piper model on disk for this host at all -- edge_voice alone is
        # enough to count as "can speak" for engine: edge_tts.
        data = dict(BASE_CONFIG)
        data["tts"] = dict(data["tts"], engine="edge_tts")
        data["dj"] = dict(data["dj"])
        data["dj"]["hosts"] = [{
            "name": "Solo", "voice_model": "/does/not/exist.onnx",
            "edge_voice": "en-US-AriaNeural"}]
        tts = TTS(Config(data, self.tmp))
        self.assertEqual(tts.hosts, ["Solo"])

    def test_a_host_with_only_a_piper_model_can_still_speak(self):
        # No edge voice configured -- still speaks, just always via fallback.
        voice = self.cfg.path("state") / "fallback-only.onnx"
        voice.write_bytes(b"not really a model, but it exists")
        data = dict(BASE_CONFIG)
        data["tts"] = dict(data["tts"], engine="edge_tts")
        data["dj"] = dict(data["dj"])
        data["dj"]["hosts"] = [{"name": "Solo", "voice_model": str(voice),
                                "edge_voice": ""}]
        tts = TTS(Config(data, self.tmp))
        self.assertEqual(tts.hosts, ["Solo"])

    def test_edge_voice_is_part_of_the_cache_key(self):
        base = self._tts_with(edge_voice="en-US-AriaNeural")
        other = self._tts_with(edge_voice="en-US-GuyNeural")
        self.assertNotEqual(base.voices["Solo"].key(),
                            other.voices["Solo"].key())

    def test_a_negative_rate_is_one_token_not_two(self):
        # "--rate", "-12%" as separate argv entries reads as two flags to
        # argparse (edge-tts's own CLI), since "-12%" itself starts with "-".
        # "--rate=-12%" sidesteps that ambiguity entirely.
        tts = self._tts_with(edge_rate="-12%", edge_pitch="-5Hz")
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            out = _output_path(command)
            if out is not None:
                out.write_bytes(b"x" * 2000)
            return _FakeCompleted(0)

        tts_module, originals = self._patched(
            {"edge-tts": "/usr/bin/edge-tts", "ffmpeg": "/usr/bin/ffmpeg"},
            fake_run)
        try:
            result = tts._synthesize("hello", self.tmp / "out.wav",
                                     tts.voices["Solo"])
        finally:
            self._restore(tts_module, originals)

        self.assertIsNotNone(result)
        edge_command = calls[0]
        self.assertIn("--rate=-12%", edge_command)
        self.assertIn("--pitch=-5Hz", edge_command)
        self.assertNotIn("-12%", edge_command, "must not be its own argv entry")
        self.assertNotIn("-5Hz", edge_command, "must not be its own argv entry")


class TestStitchingAudio(RadioTestCase):
    """The conversation splicer, which is the one bit of new audio handling.

    _join_wavs is pure stdlib, so it runs everywhere. The ffmpeg path is the
    one the laptop actually uses and is checked too when ffmpeg is around.
    """

    def setUp(self):
        super().setUp()
        self.tts = TTS(self.cfg)

    def _parts(self, lengths=(0.4, 0.3), rate=22050):
        made = []
        for index, seconds in enumerate(lengths):
            path = self.tmp / f"part{index}.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(
                    struct.pack("<h", 3000 + index * 900) * int(rate * seconds))
            made.append(path)
        return made

    def test_turns_are_spliced_end_to_end_with_a_gap(self):
        parts = self._parts((0.4, 0.3))
        target = self.tmp / "joined.wav"
        self.assertTrue(self.tts._join_wavs(parts, target, gap=0.35))
        # 0.4 + 0.35 of silence + 0.3, give or take a frame.
        self.assertAlmostEqual(wav_duration(target), 1.05, places=2)

    def test_splicing_keeps_the_format(self):
        parts = self._parts((0.2, 0.2), rate=22050)
        target = self.tmp / "joined.wav"
        self.tts._join_wavs(parts, target, gap=0.1)
        with wave.open(str(target), "rb") as handle:
            self.assertEqual(handle.getframerate(), 22050)
            self.assertEqual(handle.getnchannels(), 1)

    def test_mismatched_voices_are_refused_rather_than_mangled(self):
        # Two Piper models can differ in sample rate. Splicing them by hand
        # would play the second voice at the wrong pitch.
        first = self._parts((0.3,), rate=22050)[0]
        second = self.tmp / "other.wav"
        with wave.open(str(second), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(struct.pack("<h", 2000) * 4800)
        self.assertFalse(
            self.tts._join_wavs([first, second], self.tmp / "out.wav", gap=0.2))

    def test_a_single_turn_still_produces_a_file(self):
        parts = self._parts((0.5,))
        target = self.tmp / "one.wav"
        self.assertTrue(self.tts._join_wavs(parts, target, gap=0.0))
        self.assertAlmostEqual(wav_duration(target), 0.5, places=2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_ffmpeg_joins_and_normalises_in_one_pass(self):
        parts = self._parts((0.4, 0.3))
        target = self.tmp / "ff.wav"
        self.assertTrue(self.tts._postprocess(parts, target, gap=0.35))
        self.assertAlmostEqual(wav_duration(target), 1.05, places=1)
        with wave.open(str(target), "rb") as handle:
            # Everything reaching the stream is resampled to match the music.
            self.assertEqual(handle.getframerate(), 44100)
            self.assertEqual(handle.getnchannels(), 2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_ffmpeg_handles_voices_that_disagree_on_sample_rate(self):
        first = self._parts((0.3,), rate=22050)[0]
        second = self.tmp / "other.wav"
        with wave.open(str(second), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(struct.pack("<h", 2000) * 4800)
        self.assertTrue(
            self.tts._postprocess([first, second], self.tmp / "out.wav", gap=0.2))


# -- the website feed -------------------------------------------------------


class TestSitePublisher(RadioTestCase):
    def setUp(self):
        super().setUp()
        self.seed_library(artists=3, per_artist=3)
        self.site = SitePublisher(self.cfg, self.db, self.queue, self.library,
                                  FakeWeather(dict(SAMPLE_READING)))

    def _queue_a_block(self):
        tracks = [dict(row) for row in self.db.query("SELECT * FROM tracks LIMIT 3")]
        self.queue.build_block({"items": [
            {"kind": "song", "track": tracks[0]},
            {"kind": "banter", "lines": [{"host": "Ray", "text": "Hi."},
                                         {"host": "Nina", "text": "Hello."}]},
            {"kind": "song", "track": tracks[1]},
        ], "source": "llm", "mood_name": "night", "show_note": "A quiet one."})
        return tracks

    def test_payload_has_everything_the_site_renders(self):
        self._queue_a_block()
        document = self.site.payload("Artist 0 - Track 0", since=123.0)
        for key in ("updated_at", "station", "now_playing", "show", "queue",
                    "history", "library", "weather"):
            self.assertIn(key, document)
        self.assertEqual(document["station"]["hosts"], ["Ray", "Nina"])
        self.assertEqual(document["show"]["note"], "A quiet one.")
        self.assertEqual(document["library"]["tracks"], 9)

    def test_now_playing_is_split_into_fields(self):
        self._queue_a_block()
        current = self.site.payload("Artist 1 - Track 2", since=99.0)["now_playing"]
        self.assertEqual(current["artist"], "Artist 1")
        self.assertEqual(current["title"], "Track 2")
        self.assertEqual(current["kind"], "song")
        self.assertEqual(current["started_at"], 99.0)
        self.assertGreater(current["duration"], 0)

    def test_patter_on_air_is_labelled_not_guessed_at(self):
        current = self.site.payload("Private Radio - Station ID")["now_playing"]
        self.assertEqual(current["kind"], "patter")

    def test_an_unknown_string_still_produces_something_showable(self):
        current = self.site.payload("Some Bootleg - A Track")["now_playing"]
        self.assertEqual(current["artist"], "Some Bootleg")
        self.assertEqual(current["title"], "A Track")

    def test_the_queue_shows_conversations_without_leaking_the_script(self):
        self._queue_a_block()
        queue = self.site.payload()["queue"]
        talk = [item for item in queue if item["kind"] == "banter"]
        self.assertEqual(len(talk), 1)
        self.assertEqual(talk[0]["hosts"], "Ray & Nina")
        # What the hosts are about to say is a spoiler, so it is not published.
        self.assertNotIn("text", talk[0])

    def test_publishing_is_off_until_it_is_configured(self):
        self.assertFalse(self.site.enabled)
        self.assertIn("site.enabled", self.site.why_disabled())
        self.assertFalse(self.site.maybe_publish("Artist 0 - Track 0"))

    def test_an_unchanged_station_is_not_republished(self):
        self._queue_a_block()
        self.site._configured = True
        self.site.token = "x"
        self.site.gist_id = "y"
        sent = []
        self.site._write = lambda document: (sent.append(document), True)[1]

        self.assertTrue(self.site.maybe_publish("Artist 0 - Track 0"))
        self.site._last_publish = 0.0          # pretend the interval elapsed
        self.assertFalse(self.site.maybe_publish("Artist 0 - Track 0"))
        self.site._last_publish = 0.0
        self.assertTrue(self.site.maybe_publish("Artist 1 - Track 1"))
        self.assertEqual(len(sent), 2)

    def test_a_failed_push_backs_off_instead_of_hammering_github(self):
        self.site._configured = True
        self.site.token = "x"
        self.site.gist_id = "y"
        self.site._session.patch = _raise
        self.assertFalse(self.site.publish_now({"a": 1}))
        self.assertEqual(self.site._failures, 1)
        self.assertGreater(self.site._quiet_until, time.time())

    def test_create_gist_surfaces_githubs_own_error_message(self):
        # A bare requests.HTTPError only says "401 Client Error: Unauthorized
        # for url: ...", which is useless for telling "bad token" apart from
        # "wrong scope" apart from "fine-grained tokens can't do this at all".
        # GitHub's response body says which one it actually is.
        self.site.token = "bad-token"

        class FakeResponse:
            ok = False
            status_code = 401
            reason = "Unauthorized"

            def json(self):
                return {"message": "Bad credentials"}

        self.site._session.post = lambda *a, **k: FakeResponse()
        with self.assertRaises(RuntimeError) as ctx:
            self.site.create_gist()
        self.assertIn("401", str(ctx.exception))
        self.assertIn("Bad credentials", str(ctx.exception))

    def test_create_gist_works_without_a_json_body(self):
        # Some failure modes (a proxy, a rate limit) return no JSON at all.
        # The message must fall back to the HTTP reason instead of crashing.
        self.site.token = "bad-token"

        class FakeResponse:
            ok = False
            status_code = 403
            reason = "Forbidden"

            def json(self):
                raise ValueError("not json")

        self.site._session.post = lambda *a, **k: FakeResponse()
        with self.assertRaises(RuntimeError) as ctx:
            self.site.create_gist()
        self.assertIn("403", str(ctx.exception))
        self.assertIn("Forbidden", str(ctx.exception))

    def test_create_gist_succeeds_and_returns_id_and_url(self):
        self.site.token = "good-token"

        class FakeResponse:
            ok = True

            def json(self):
                return {"id": "abc123", "html_url": "https://gist.github.com/abc123"}

        self.site._session.post = lambda *a, **k: FakeResponse()
        gist_id, url = self.site.create_gist()
        self.assertEqual(gist_id, "abc123")
        self.assertEqual(url, "https://gist.github.com/abc123")

    def test_create_gist_refuses_without_a_token(self):
        self.site.token = ""
        with self.assertRaises(RuntimeError):
            self.site.create_gist()


def _raise(*args, **kwargs):
    raise RuntimeError("github is down")


# -- schema migration -------------------------------------------------------


class TestMigrations(RadioTestCase):
    def test_an_older_database_gains_the_host_column(self):
        path = self.tmp / "state" / "old.db"
        connection = sqlite3.connect(path)
        connection.executescript(
            "CREATE TABLE queue_items (id INTEGER PRIMARY KEY, block_id INTEGER,"
            " seq INTEGER NOT NULL, kind TEXT NOT NULL, tier TEXT NOT NULL"
            " DEFAULT 'ai', path TEXT, track_id INTEGER, text TEXT, title TEXT,"
            " artist TEXT, duration REAL, status TEXT NOT NULL DEFAULT 'pending',"
            " created_at REAL NOT NULL, pushed_at REAL);"
            "INSERT INTO queue_items(seq, kind, created_at) VALUES(1,'patter',0);")
        connection.commit()
        connection.close()

        db = Database(path)
        db.init()
        columns = {row["name"] for row in db.query("PRAGMA table_info(queue_items)")}
        self.assertIn("host", columns)
        # The row that was already there survives, with an empty host.
        row = db.one("SELECT * FROM queue_items")
        self.assertIsNone(row["host"])

    def test_migrating_twice_is_harmless(self):
        Database(self.cfg.db_path).init()
        Database(self.cfg.db_path).init()
        columns = {row["name"] for row in
                   self.db.query("PRAGMA table_info(queue_items)")}
        self.assertIn("host", columns)


if __name__ == "__main__":
    unittest.main()
