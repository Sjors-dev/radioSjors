"""Command line entry point: airadio <command>."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import Config, get_config
from .db import Database
from .discovery import LastFM
from .downloader import Downloader
from .library import Library
from .liquidsoap import AI_QUEUE, REQUEST_QUEUE, LiquidsoapClient
from .publisher import SitePublisher
from .queueing import QueueManager
from .tts import TTS
from .util import human_duration, setup_logging
from .weather import Weather
from .brain.llm import LLM, LLMUnavailable
from .brain.planner import Planner


# -- shared wiring ----------------------------------------------------------


class App:
    """Everything the CLI needs, built once."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = Database(cfg.db_path)
        self.db.init()
        self.library = Library(self.db, cfg.path("library"))
        self.lastfm = LastFM(Config.env("LASTFM_API_KEY"))
        self.llm = LLM(cfg)
        self.tts = TTS(cfg)
        self.weather = Weather(cfg)
        self.planner = Planner(cfg, self.db, self.library, self.llm, self.weather)
        self.downloader = Downloader(cfg, self.db, self.library, self.lastfm)
        self.queue = QueueManager(cfg, self.db, self.library, self.tts)
        self.site = SitePublisher(cfg, self.db, self.queue, self.library,
                                  self.weather)
        self.ls = LiquidsoapClient(
            str(cfg.get("stream.telnet_host", "127.0.0.1")),
            int(cfg.get("stream.telnet_port", 1234)),
        )


# -- commands ---------------------------------------------------------------


def cmd_init(args, cfg: Config) -> int:
    for key in ("library", "banned", "queue", "state", "logs"):
        path = cfg.path(key)
        print(f"  {key:<8} {path}")
    App(cfg)
    print(f"\nDatabase ready: {cfg.db_path}")
    if not (cfg.root / ".env").exists():
        print("\n!! No .env yet. Copy .env.example to .env and fill it in.")
    return 0


def cmd_scan(args, cfg: Config) -> int:
    app = App(cfg)
    summary = app.library.scan()
    print(f"added {summary['added']}, updated {summary['updated']}, "
          f"duplicates skipped {summary['skipped']}, "
          f"missing marked {summary['pruned']}")
    print(f"library now holds {summary['total']} tracks")
    return 0


def cmd_stream_config(args, cfg: Config) -> int:
    template_path = cfg.root / "stream" / "radio.liq.template"
    target = cfg.root / "stream" / "radio.generated.liq"
    template = template_path.read_text(encoding="utf-8")

    values = {
        "TELNET_HOST": str(cfg.get("stream.telnet_host", "127.0.0.1")),
        "TELNET_PORT": str(cfg.get("stream.telnet_port", 1234)),
        "SAMPLERATE": str(cfg.get("station.samplerate", 44100)),
        "BITRATE": str(cfg.get("station.bitrate", 128)),
        "CROSSFADE": f"{float(cfg.get('stream.crossfade_seconds', 2.0)):.1f}",
        "LIBRARY_DIR": str(cfg.path("library")).replace("\\", "/"),
        "ICECAST_HOST": Config.env("ICECAST_HOST", "localhost"),
        "ICECAST_PORT": Config.env("ICECAST_PORT", "8000"),
        "ICECAST_MOUNT": Config.env("ICECAST_MOUNT", "/stream"),
        "ICECAST_USER": Config.env("ICECAST_USER", "source"),
        "ICECAST_PASSWORD": Config.env("ICECAST_PASSWORD", "hackme"),
        "STATION_NAME": str(cfg.get("station.name", "Private Radio")),
        "STATION_DESC": str(cfg.get("station.description", "")),
        "STATION_GENRE": str(cfg.get("station.genre", "Various")),
    }
    for key, value in values.items():
        template = template.replace(f"@{key}@", value)

    if "@" in template and any(f"@{k}@" in template for k in values):
        print("!! some placeholders were not substituted", file=sys.stderr)

    target.write_text(template, encoding="utf-8")
    if os.name == "posix":
        os.chmod(target, 0o600)  # it holds the caster.fm password
    print(f"wrote {target}")

    if shutil.which("liquidsoap"):
        result = subprocess.run(["liquidsoap", "--check", str(target)],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("liquidsoap --check: OK")
        else:
            print("liquidsoap --check FAILED:\n" + (result.stderr or result.stdout))
            return 1
    else:
        print("(liquidsoap not installed here, skipping syntax check)")
    return 0


def cmd_bootstrap(args, cfg: Config) -> int:
    app = App(cfg)
    if not app.lastfm.enabled:
        print("LASTFM_API_KEY is not set - cannot discover anything.", file=sys.stderr)
        return 1

    target = args.count
    print(f"growing the library to {target} tracks "
          f"(currently {app.library.count()}); one download at a time")

    # Seeded in batches. Each pass is a run of throttled Last.fm calls, and the
    # loop re-seeds whenever it runs dry, so asking for hundreds up front only
    # buys minutes of silence before the first download.
    print("  finding candidates on Last.fm...", flush=True)
    app.downloader.seed_candidates(wanted=60)
    failures = 0
    waits = 0
    while app.library.count() < target and failures < 15:
        result = app.downloader.fetch_next_candidate()
        if result is None:
            print("  finding more candidates...", flush=True)
            if app.downloader.seed_candidates(wanted=60) == 0:
                print("ran out of candidates from Last.fm")
                break
            continue
        if result.busy:
            # The brain is mid-download. That is not a failure of anything, so
            # wait for it rather than spending the failure budget losing a
            # lock race fifteen times in a row.
            waits += 1
            if waits == 1 or waits % 10 == 0:
                print("  waiting for the brain to finish its download...",
                      flush=True)
            if waits > 240:
                print("the brain has held the download lock for a long time; "
                      "stop ai-radio-brain and re-run, or lower "
                      "discovery.downloads_per_hour")
                break
            time.sleep(5)
            continue
        waits = 0
        if result.ok:
            failures = 0
            print(f"  [{app.library.count():>3}] "
                  f"{result.track['artist']} - {result.track['title']}")
        else:
            failures += 1
            print(f"  skipped: {result.reason}")

    print(f"library now holds {app.library.count()} tracks")
    return 0


def cmd_plan(args, cfg: Config) -> int:
    app = App(cfg)
    if app.library.count() == 0:
        print("library is empty - run 'airadio bootstrap' first", file=sys.stderr)
        return 1

    plan = app.planner.plan_block(airs_in=app.queue.ready_seconds(),
                                  speaking_hosts=app.tts.hosts or None)
    print(f"source: {plan['source']}   slot: {plan['mood_name']}")
    if plan.get("show_note"):
        print(f"note:   {plan['show_note']}")
    print()
    for item in plan["items"]:
        if item["kind"] == "song":
            track = item["track"]
            print(f"  [song]   {track['artist']} - {track['title']} "
                  f"({human_duration(track.get('duration'))})")
        elif item["kind"] == "banter":
            print("  [banter]")
            for turn in item["lines"]:
                print(f"    {turn['host']}: {turn['text']}")
        else:
            who = item.get("host") or ""
            print(f"  [patter] {who + ': ' if who else ''}{item['text']}")

    if args.dry_run:
        print("\n(dry run - nothing queued, no TTS rendered)")
        return 0

    count = app.queue.build_block(plan)
    print(f"\nqueued {count} items; buffer is now "
          f"{human_duration(app.queue.ready_seconds())}")
    return 0


def cmd_run(args, cfg: Config) -> int:
    from .runner import Runner

    setup_logging(cfg.path("logs"), "brain", args.log_level)
    Runner(cfg).run()
    return 0


def cmd_bot(args, cfg: Config) -> int:
    from .bot.discord_bot import run as run_bot

    setup_logging(cfg.path("logs"), "bot", args.log_level)
    run_bot(cfg)
    return 0


def cmd_download(args, cfg: Config) -> int:
    app = App(cfg)
    result = app.downloader.fetch(args.artist, args.title,
                                  allow_remix=args.allow_remix)
    if result.ok:
        print(f"OK: {result.track['path']}")
        return 0
    print(f"FAILED: {result.reason}", file=sys.stderr)
    return 1


def cmd_tts_test(args, cfg: Config) -> int:
    app = App(cfg)
    ok, detail = app.tts.check()
    print(f"engine check: {detail}")
    if not ok:
        return 1

    if args.duet:
        # Prove the two voices actually stitch together before trusting a
        # whole hour to it.
        names = app.tts.hosts
        if len(names) < 2:
            print(f"need two voices for a duet: {app.tts.check_exchange()[1]}",
                  file=sys.stderr)
            return 1
        lines = [{"host": names[index % 2], "text": part.strip()}
                 for index, part in enumerate(args.text.split("|"))
                 if part.strip()]
        if len(lines) < 2:
            lines = [{"host": names[0], "text": args.text},
                     {"host": names[1], "text": "And that is the other one."}]
        path = app.tts.render_exchange(lines, name_hint="test")
    else:
        path = app.tts.render(args.text, name_hint="test", host=args.host)

    if path is None:
        print("render failed", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


def cmd_weather(args, cfg: Config) -> int:
    weather = Weather(cfg)
    if not weather.enabled:
        print("weather is disabled in config (weather.enabled)")
        return 0
    reading = weather.current(force=True)
    if not reading:
        print("could not reach Open-Meteo", file=sys.stderr)
        return 1
    for key, value in reading.items():
        if key != "fetched_at":
            print(f"  {key:<12} {value}")
    print(f"\nas the hosts are told it:\n  {weather.briefing()}")
    return 0


def cmd_site_init(args, cfg: Config) -> int:
    """Create the gist the website reads, once."""
    app = App(cfg)
    if not Config.env("GITHUB_TOKEN"):
        print("Set GITHUB_TOKEN in .env first.\n"
              "Make one at https://github.com/settings/tokens -> "
              "'Generate new token (classic)', tick ONLY the 'gist' scope.",
              file=sys.stderr)
        return 1
    if Config.env("SITE_GIST_ID") and not args.force:
        print(f"SITE_GIST_ID is already set ({Config.env('SITE_GIST_ID')}).\n"
              "Pass --force to create another one.")
        return 0
    try:
        gist_id, url = app.site.create_gist()
    except Exception as exc:
        print(f"could not create the gist: {exc}", file=sys.stderr)
        return 1
    print(f"Created: {url}\n")
    print("Add this to .env on the laptop:")
    print(f"  SITE_GIST_ID={gist_id}")
    print("\nand set these in Vercel's environment variables:")
    print(f"  GIST_ID={gist_id}")
    print("  GITHUB_TOKEN=<the same token, or a read-only one>")
    print("  SITE_PASSWORD=<whatever you want to type to get in>")
    print("\nThen set site.enabled: true in config/config.local.yaml.")
    return 0


def cmd_site_preview(args, cfg: Config) -> int:
    """Print exactly what the website would be told right now."""
    import json

    app = App(cfg)
    now_playing = ""
    try:
        now_playing = cfg.now_playing_path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    document = app.site.payload(now_playing, time.time())
    print(json.dumps(document, indent=2, ensure_ascii=False, default=str))
    if not app.site.enabled:
        print(f"\n(not publishing: {app.site.why_disabled()})", file=sys.stderr)
    elif args.publish:
        ok = app.site.publish_now(document)
        print(f"\npublished: {ok}", file=sys.stderr)
    return 0


def cmd_mood(args, cfg: Config) -> int:
    app = App(cfg)
    if args.clear:
        app.planner.set_mood("")
        print("mood cleared - back to the time-of-day map")
        return 0
    if args.text:
        app.planner.set_mood(" ".join(args.text))
    print(f"current mood: {app.planner.current_mood() or '(time of day only)'}")
    return 0


def cmd_status(args, cfg: Config) -> int:
    app = App(cfg)
    tracks = app.library.count()
    ready = app.queue.ready_seconds()
    pending = app.db.one("SELECT COUNT(*) AS n FROM candidates WHERE status='new'")
    connected = app.ls.connected

    print(f"library        {tracks} tracks")
    print(f"buffer ready   {human_duration(ready)} "
          f"({app.queue.ready_count()} items waiting)")
    print(f"mood           {app.planner.current_mood() or '(time of day only)'}")
    print(f"llm providers  {', '.join(app.llm.available_providers) or 'none'}")
    print(f"last.fm        {'enabled' if app.lastfm.enabled else 'no key'}")
    print(f"download queue {pending['n'] if pending else 0} candidates")
    print(f"liquidsoap     {'connected' if connected else 'not reachable'}")
    if connected:
        print(f"  ai queue     {app.ls.queue_length(AI_QUEUE)} items pushed")
        print(f"  request queue {app.ls.queue_length(REQUEST_QUEUE)} items pushed")
        print(f"  now playing  {app.ls.now_playing() or 'unknown'}")

    print("\nupcoming:")
    for item in app.queue.preview(8):
        label = item["title"] if item["kind"] == "song" else "[patter]"
        who = item["artist"] or ""
        mark = "*" if item["tier"] == "request" else " "
        print(f" {mark} {who} - {label}" if who else f" {mark} {label}")
    return 0


def cmd_banned(args, cfg: Config) -> int:
    app = App(cfg)
    rows = app.db.query("SELECT * FROM banned ORDER BY created_at DESC")
    if not rows:
        print("nothing is banned")
        return 0
    for row in rows:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["created_at"]))
        location = row["file_path"] or "(file was already gone)"
        print(f"{stamp}  {row['artist']} - {row['title']}")
        print(f"                  key: {row['dedupe_key']}")
        print(f"                  {location}")
    print(f"\n{len(rows)} banned. Restore one with: "
          f"main.py unban \"<artist>\" \"<title>\"")
    return 0


def cmd_unban(args, cfg: Config) -> int:
    from .util import dedupe_key

    app = App(cfg)
    key = dedupe_key(args.artist, args.title)
    if not app.library.unban(key, library_dir=cfg.path("library")):
        print(f"not banned: {args.artist} - {args.title}", file=sys.stderr)
        return 1
    app.library.scan()
    print(f"restored {args.artist} - {args.title}")
    return 0


def cmd_log(args, cfg: Config) -> int:
    app = App(cfg)
    rows = app.db.query(
        "SELECT * FROM download_log ORDER BY id DESC LIMIT ?", (args.count,))
    for row in rows:
        stamp = time.strftime("%m-%d %H:%M", time.localtime(row["created_at"]))
        print(f"{stamp}  {row['decision']:<9} {row['artist']} - {row['title']}")
        print(f"            {row['reason']}")
        if row["video_title"]:
            print(f"            candidate: {row['video_title']} [{row['channel']}]")
    return 0


def cmd_doctor(args, cfg: Config) -> int:
    """Check every moving part and say plainly what is not ready."""
    problems = 0
    warnings = 0

    def report(status: str, label: str, detail: str = "") -> None:
        nonlocal problems, warnings
        if status == "FAIL":
            problems += 1
        elif status == "WARN":
            warnings += 1
        line = f"[{status:^4}] {label}"
        print(f"{line}\n         {detail}" if detail else line)

    print("=== paths ===")
    for key in ("library", "banned", "queue", "state", "logs"):
        path = cfg.path(key)
        writable = os.access(path, os.W_OK)
        report("OK" if writable else "FAIL", f"{key} dir", str(path))

    print("\n=== external tools ===")
    for tool, required, hint in (
        ("ffmpeg", True, "sudo apt install ffmpeg"),
        ("liquidsoap", True, "sudo apt install liquidsoap"),
    ):
        found = shutil.which(tool)
        report("OK" if found else "FAIL", tool, found or f"missing - {hint}")

    try:
        import yt_dlp
        report("OK", "yt-dlp", f"version {yt_dlp.version.__version__}")
    except Exception as exc:
        report("FAIL", "yt-dlp", str(exc))

    print("\n=== text to speech ===")
    app = App(cfg)
    ok, detail = app.tts.check()
    report("OK" if ok else "WARN", "tts engine", detail)
    duet_ok, duet_detail = app.tts.check_exchange()
    report("OK" if duet_ok else "WARN", "two-host segments", duet_detail)

    print("\n=== credentials ===")
    for name, required in (("ICECAST_HOST", True), ("ICECAST_PASSWORD", True),
                           ("LASTFM_API_KEY", False), ("DISCORD_TOKEN", False)):
        value = Config.env(name)
        if value:
            report("OK", name, "set")
        else:
            report("FAIL" if required else "WARN", name, "not set in .env")

    providers = app.llm.available_providers
    report("OK" if providers else "WARN", "llm keys",
           ", ".join(providers) or "none set - the fallback planner will be used")

    print("\n=== live checks ===")
    if app.lastfm.enabled:
        similar = app.lastfm.similar_artists("Radiohead", limit=3)
        report("OK" if similar else "FAIL", "last.fm api",
               ", ".join(similar) or "no response")
    else:
        report("WARN", "last.fm api", "skipped, no key")

    if providers:
        try:
            answer = app.llm.complete(
                "Answer with exactly one word.", "Say the word: ready",
                temperature=0.0, max_tokens=app.llm.budget("probe"))
            report("OK", "llm call", f"answered {answer.strip()[:40]!r}")
        except LLMUnavailable as exc:
            report("FAIL", "llm call", str(exc)[:200])
    else:
        report("WARN", "llm call", "skipped, no key")

    generated = cfg.root / "stream" / "radio.generated.liq"
    if generated.exists():
        if shutil.which("liquidsoap"):
            result = subprocess.run(["liquidsoap", "--check", str(generated)],
                                    capture_output=True, text=True)
            report("OK" if result.returncode == 0 else "FAIL", "liquidsoap script",
                   "syntax ok" if result.returncode == 0
                   else (result.stderr or result.stdout)[:300])
        else:
            report("WARN", "liquidsoap script", "cannot check, liquidsoap missing")
    else:
        report("WARN", "liquidsoap script",
               "not generated yet - run: airadio stream-config")

    report("OK" if app.ls.connected else "WARN", "liquidsoap telnet",
           "connected" if app.ls.connected
           else "not running (fine if you have not started the stream yet)")

    if app.weather.enabled:
        brief = app.weather.briefing()
        report("OK" if brief else "WARN", "weather",
               brief or "Open-Meteo unreachable - the hosts will skip it")
    else:
        report("WARN", "weather", "disabled in config")

    if app.site.enabled:
        report("OK" if app.site.publish_now() else "FAIL", "website feed",
               f"gist {app.site.gist_id}")
    else:
        report("WARN", "website feed", app.site.why_disabled())

    count = app.library.count()
    minimum = int(cfg.get("discovery.min_tracks_to_program", 25))
    report("OK" if count >= minimum else "WARN", "library size",
           f"{count} tracks" + ("" if count >= minimum
                                else f" - below {minimum}, run: airadio bootstrap"))

    print(f"\n{problems} failure(s), {warnings} warning(s)")
    return 1 if problems else 0


# -- argument parsing -------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airadio", description="AI-programmed private radio station")
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create directories and the database")
    sub.add_parser("scan", help="index the library folder")
    sub.add_parser("stream-config", help="render stream/radio.generated.liq")
    sub.add_parser("doctor", help="check every dependency and credential")
    sub.add_parser("status", help="show what the station is doing")

    boot = sub.add_parser("bootstrap", help="fill an empty library from Last.fm")
    boot.add_argument("--count", type=int, default=40,
                      help="stop once the library has this many tracks")

    plan = sub.add_parser("plan", help="plan one block now")
    plan.add_argument("--dry-run", action="store_true",
                      help="show the plan without rendering or queueing it")

    sub.add_parser("run", help="run the brain loop (the main service)")
    sub.add_parser("bot", help="run the Discord bot")

    download = sub.add_parser("download", help="fetch one track by hand")
    download.add_argument("artist")
    download.add_argument("title")
    download.add_argument("--allow-remix", action="store_true")

    tts = sub.add_parser("tts-test", help="render one line of speech")
    tts.add_argument("text", nargs="?",
                     default="You're listening to the station. Here's the next one.")
    tts.add_argument("--host", help="which host says it (default: the first)")
    tts.add_argument("--duet", action="store_true",
                     help="render a two-host exchange; split turns with |")

    sub.add_parser("weather", help="show the weather the hosts are given")

    site_init = sub.add_parser("site-init",
                               help="create the gist the website reads")
    site_init.add_argument("--force", action="store_true",
                           help="make a new gist even if one is configured")

    site_preview = sub.add_parser("site-preview",
                                  help="print what the website would be told")
    site_preview.add_argument("--publish", action="store_true",
                              help="also push it to the gist right now")

    mood = sub.add_parser("mood", help="read or set the station mood")
    mood.add_argument("text", nargs="*")
    mood.add_argument("--clear", action="store_true")

    sub.add_parser("banned", help="list banned tracks")

    unban = sub.add_parser("unban", help="restore a banned track")
    unban.add_argument("artist")
    unban.add_argument("title")

    logs = sub.add_parser("log", help="recent download decisions")
    logs.add_argument("--count", type=int, default=20)

    return parser


COMMANDS = {
    "init": cmd_init,
    "scan": cmd_scan,
    "stream-config": cmd_stream_config,
    "bootstrap": cmd_bootstrap,
    "plan": cmd_plan,
    "run": cmd_run,
    "bot": cmd_bot,
    "download": cmd_download,
    "tts-test": cmd_tts_test,
    "weather": cmd_weather,
    "site-init": cmd_site_init,
    "site-preview": cmd_site_preview,
    "mood": cmd_mood,
    "status": cmd_status,
    "banned": cmd_banned,
    "unban": cmd_unban,
    "log": cmd_log,
    "doctor": cmd_doctor,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config) if args.config else get_config()

    if args.command not in ("run", "bot"):
        setup_logging(cfg.path("logs"), "cli", args.log_level)

    try:
        return COMMANDS[args.command](args, cfg)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
