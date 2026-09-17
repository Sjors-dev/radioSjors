"""Offline tests: everything that does not need ffmpeg, liquidsoap or a network.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import struct
import sys
import tempfile
import unittest
import wave
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from airadio.config import Config
from airadio.db import Database
from airadio.discovery import LastFM
from airadio.downloader import Downloader
from airadio.library import Library
from airadio.liquidsoap import annotate_uri, format_metadata, parse_metadata
from airadio.queueing import QueueManager
from airadio.tts import TTS
from airadio.util import dedupe_key, normalize, safe_filename
from airadio.brain.intent import _rules
from airadio.brain.llm import LLM, extract_json
from airadio.brain.planner import Planner

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
        self.assertTrue(all(row["kind"] == "patter" for row in pending))
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

    def test_take_next_marks_tracks_played(self):
        self.build()
        item = self.queue.take_next("ai", limit=1)[0]
        row = self.db.one("SELECT play_count, last_played_at FROM tracks WHERE id=?",
                          (item["track_id"],))
        self.assertEqual(row["play_count"], 1)
        self.assertIsNotNone(row["last_played_at"])

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


if __name__ == "__main__":
    unittest.main()
