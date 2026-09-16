# Build status

Written on the main Windows PC. The target is the ASUS X551MA on Linux Mint 22.2.

## Done — all four milestones from the brief are implemented

1. **Stream something** — `stream/radio.liq.template`, generated into
   `stream/radio.generated.liq` by `main.py stream-config`. Three-tier fallback
   plus `mksafe`, auto-reconnect to caster.fm.
2. **AI programming** — hourly planner (one LLM pass per block), Piper patter,
   queue buffered ~1h ahead, deterministic fallback planner when the LLM is out.
3. **Self-growing library** — Last.fm discovery, yt-dlp downloads with quality
   filters, background fetch throttled to one download at a time.
4. **Live requests** — Discord bot, intent classification (specific request vs
   vibe shift), request tier and shared mood state.

Plus: systemd units, installer scripts, headless setup, `doctor`, and 54 offline
tests (`.venv/bin/python -m unittest discover -s tests`), all passing.

## What was verified on this PC

- All 54 tests pass on Python 3.14.
- Full offline run-through against a synthetic 40-track library: scan → plan →
  render → queue → status, plus simulated Discord requests for all three intent
  kinds (vibe shift, track request, question) through the real `Runner`.
- Graceful degradation confirmed: with no LLM keys the fallback planner runs;
  with no TTS the patter is dropped and music continues.
- `doctor` correctly reports every missing dependency and credential.

## What could NOT be verified here, and needs a look on the laptop

These are the only genuinely untested paths. All are caught by step 3 and 4 of
SETUP.md, so nothing here is a silent failure.

1. **The liquidsoap script has never been run.** There is no liquidsoap build
   for Windows, and no ffmpeg on this PC. The script is written against
   Liquidsoap 2.2.x (what Ubuntu 24.04 / Mint 22.2 ships). `main.py
   stream-config` runs `liquidsoap --check` automatically and will print any
   syntax error with a line number. **This is the first thing to confirm on the
   laptop.** If a function signature differs on the installed version, the fix
   is local to `stream/radio.liq.template`.
2. **Live API calls** — Last.fm, Gemini, Groq, Discord. Request/response shapes
   are written to each provider's documented contract, and the failover logic is
   unit-tested against mocked responses, but no real key exists here.
   `main.py doctor` makes a real call to Last.fm and to the LLM, so a wrong
   payload shape shows up immediately.
3. **Piper on a no-AVX CPU.** The known risk from the brief. If onnxruntime
   faults on the Celeron N2840, `doctor` says so and `tts.engine: espeak` in
   `config/config.local.yaml` is the fallback. The station is designed to run
   fine either way.
4. **An actual yt-dlp download.** The quality-filter scoring is unit-tested
   against synthetic candidates, but no video has been fetched. Test with
   `main.py download "Khruangbin" "August 10"` before running `bootstrap`.

## Known deviations from the brief's suggested layout

- The Python package is `airadio/` rather than loose top-level `library/`,
  `queue/`, `bot/` folders. `queue` in particular would shadow the stdlib
  `queue` module. The brief's directories exist as modules inside it.
- The runner is single threaded rather than using explicit concurrency limits.
  That *is* the concurrency cap the brief asks for on a two-core CPU: a
  download, a TTS render and a re-plan cannot overlap by construction.

## If picking this up fresh

```bash
cd E:\random\customradio
.venv\Scripts\python.exe -m unittest discover -s tests   # 54 tests, should pass
.venv\Scripts\python.exe main.py doctor                  # expects ffmpeg/liquidsoap to fail on Windows
```

Then read SETUP.md, which is the laptop-side walkthrough.
