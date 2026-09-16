# AI Radio

A private, always-on internet radio station streamed to caster.fm and
programmed by an AI "DJ". It plans the music around the time of day and your
taste, voices short DJ patter between songs with local text-to-speech, grows its
own music library on demand, and takes live requests over Discord.

Single listener. Not a public broadcast.

**Start here: [SETUP.md](SETUP.md)** — the step-by-step install on the laptop.

---

## How it works

```
                  ┌─────────────┐     Last.fm API (free)
                  │   BRAIN     │────▶ discover real similar tracks
  Discord bot ───▶│  (Python)   │     yt-dlp ─▶ download + tag what's missing
                  │   planner   │
                  │             │────▶ writes all the DJ patter for the hour
                  └──────┬──────┘
                         │ block: [patter, song, patter, song, …]
                         ▼
                  ┌─────────────┐
                  │ Piper TTS   │  patter text ─▶ .wav on disk
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │   QUEUE     │  SQLite, buffered ~1h ahead
                  └──────┬──────┘
                         │ telnet push
                         ▼
                  ┌─────────────┐
                  │ LIQUIDSOAP  │  3-tier fallback ─▶ caster.fm (Icecast)
                  └─────────────┘
```

The whole design comes from one rule: **the slow AI must never be able to stop
the audio.** LLM calls, TTS renders and downloads are slow and fail sometimes.
So the brain works an hour ahead and writes finished files into a queue.
Liquidsoap only ever plays files that already exist, and Python is never in the
audio path — it pushes into liquidsoap's own queues over a local telnet socket
and then gets out of the way.

### Three tiers, highest priority first

| Tier | Source | Fires when |
|------|--------|-----------|
| 1 | **Request queue** | You asked for a specific track in Discord. Plays after the current track ends. |
| 2 | **AI-planned queue** | Normal programming: the hourly block of patter and songs. |
| 3 | **Safety playlist** | Anything above is empty. Plain shuffle of the library — no Python, no network, no AI. |

Below all three, liquidsoap's `mksafe` streams silence rather than dropping the
mount, so caster.fm never sees the source disappear.

Kill the brain and the music keeps playing. That is the point, and
[SETUP.md](SETUP.md#8-prove-it-never-goes-silent) has the test for it.

---

## What each piece does

| Path | Job |
|------|-----|
| `airadio/config.py` | `config.yaml` for behaviour, `.env` for secrets, `config.local.yaml` for machine overrides |
| `airadio/db.py` | SQLite schema and access. WAL mode — three processes share it |
| `airadio/library.py` | Scan the music folder, read/write tags, de-duplicate by tags *and* by content hash, ban/unban |
| `airadio/discovery.py` | Last.fm client: real similar artists and tracks, plus reference durations |
| `airadio/downloader.py` | yt-dlp wrapper and the quality filters that make it usable |
| `airadio/tts.py` | Piper (or espeak) renderer, loudness-matched to the music |
| `airadio/brain/llm.py` | Gemini ⇄ Groq with failover and rate-limit cooldown |
| `airadio/brain/planner.py` | The hourly block: one LLM pass picks the order and writes every patter line. Deterministic fallback when the LLM is down |
| `airadio/brain/intent.py` | Chat message → specific request / vibe shift / question |
| `airadio/queueing.py` | Renders patter, verifies files, keeps ~1h of ready audio |
| `airadio/liquidsoap.py` | Telnet client: push, queue depth, now-playing, skip |
| `airadio/runner.py` | The brain loop. Single threaded on purpose — that *is* the concurrency cap |
| `airadio/bot/discord_bot.py` | Discord front end. Writes to SQLite, never does slow work itself |
| `stream/radio.liq.template` | The liquidsoap script. `airadio stream-config` fills in the secrets |

---

## Commands

```bash
.venv/bin/python main.py doctor           # check every dependency and credential
.venv/bin/python main.py status           # library, buffer, mood, now playing
.venv/bin/python main.py bootstrap -h     # fill an empty library from Last.fm
.venv/bin/python main.py plan --dry-run   # show the next hour without queueing it
.venv/bin/python main.py scan             # re-index the music folder
.venv/bin/python main.py download "Artist" "Title"
.venv/bin/python main.py tts-test "line to speak"
.venv/bin/python main.py mood "darker and slower"
.venv/bin/python main.py banned           # what's blacklisted
.venv/bin/python main.py unban "Artist" "Title"
.venv/bin/python main.py log              # what got downloaded, what got rejected and why
.venv/bin/python main.py run              # the brain loop (normally systemd's job)
.venv/bin/python main.py bot              # the Discord bot (normally systemd's job)
```

## Talking to it on Discord

Plain language works — the LLM classifies intent, and a regex fallback covers
it when the LLM is down:

- `play bohemian rhapsody by queen` → queued as a request; downloaded first if
  it isn't in the library
- `make it darker and slower` → updates the station mood and re-plans
- `skip this song please` / `next song` → skips the current track (does **not** ban it)
- `never play this again` / `delete this song` → bans whatever is on air
- `ban Runaway by Kanye West` → bans a named track
- `what's playing?` → now playing plus what's next

| Command | Does |
|---------|------|
| `!np` | what is on air right now |
| `!queue` | the next few tracks |
| `!status` | library size, buffer, mood, stream health |
| `!skip` | skip the current track |
| `!mood` | show the current mood |
| `!mood <text>` | set it, e.g. `!mood darker and slower` |
| `!mood reset` | back to the time-of-day schedule |
| `!ban` | ban whatever is playing now |
| `!ban <artist> - <title>` | ban a named track |
| `!banned` | list what is banned |
| `!help` | the list, in chat |

Banning moves the audio to `banned/` rather than deleting it, marks it in the
database, drops it from anything already queued, skips it if it's on air, and
stops the downloader ever fetching it again. The file has to leave `library/`
because the safety playlist reads that folder directly. Reversible with
`main.py unban "<artist>" "<title>"`.

---

## Running costs

Nothing. Last.fm, Gemini, Groq, Discord and yt-dlp are all free tiers with no
card, and the hourly-batched design makes only a handful of LLM calls per hour.
TTS is local. The only bill is caster.fm, if you choose a paid tier there.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

79 offline tests — no network, no ffmpeg, no liquidsoap needed. They cover the
quality filters, de-duplication, banning, the fallback planner, artist spacing,
queue claiming, LLM failover, and the liquidsoap telnet protocol.
