# Setup — ASUS X551MA, Linux Mint 22.2

Follow these in order. Each step is checkable before you move on.

Everything runs as your normal user. Nothing here needs a root shell; the
scripts call `sudo` where they need it.

---

## 0. Get the code onto the laptop

Copy this folder over (USB stick, `scp`, git — whatever you prefer) to
somewhere like `/home/<you>/ai-radio`, then:

```bash
cd ~/ai-radio
```

---

## 1. Collect the credentials

You need these before anything will work. All free, none need a card.

| What | Where | Notes |
|------|-------|-------|
| **caster.fm** host, port, mount, password | caster.fm dashboard → server / source details | The mount usually looks like `/stream`. Also check **which plan tier** you're on — see [note below](#a-note-on-the-castorfm-tier). |
| **Last.fm API key** | https://www.last.fm/api/account/create | Instant. Only the API key is needed, not the secret. |
| **Gemini API key** | https://aistudio.google.com/apikey | Free tier, no card. |
| **Groq API key** | https://console.groq.com/keys | Free tier, no card. Having both gives you automatic failover. |
| **Discord bot token** | https://discord.com/developers/applications | New Application → Bot → Reset Token. **Turn on "MESSAGE CONTENT INTENT"** under Bot → Privileged Gateway Intents, or the bot will see empty messages. |
| **Discord channel ID** | In Discord: Settings → Advanced → Developer Mode, then right-click your channel → Copy Channel ID | Optional. Leave blank to let the bot listen anywhere it can see. |

To get the bot into your server: Developer Portal → OAuth2 → URL Generator →
scopes `bot`, permissions `Send Messages` + `Read Message History` + `Add
Reactions`. Open the generated URL and pick your server.

---

## 2. Install

```bash
bash scripts/install.sh
```

This installs liquidsoap, ffmpeg and a Python venv, installs Piper, downloads
the `en_US-amy-medium` voice into `voices/`, and writes
`config/config.local.yaml` pointing at it.

It is safe to re-run.

> **If Piper won't run on this CPU:** the Celeron N2840 has no AVX, and some
> onnxruntime builds fault on such CPUs. `doctor` (step 4) will tell you. If it
> fails, set `tts.engine: espeak` in `config/config.local.yaml` — robotic, but
> it runs on anything, and the station still works. You can also set
> `tts.engine: none` for music with no patter at all.

> **Want more expressive voices than Piper can manage?** Set
> `tts.engine: edge_tts` and `tts.edge_voice` (and per-host `edge_voice`) in
> `config/config.local.yaml` to use Microsoft Edge's free neural voices —
> still no bill, no key, genuinely more human than Piper — via the
> `edge-tts` package `install.sh` already installed. It's an unofficial
> endpoint though, not a real product with a support line, so every host
> keeps its `voice_model` too: a failed edge-tts request (offline,
> rate-limited, blocked) falls straight back to Piper for that one line, and
> `doctor` will say plainly if it's currently running on the fallback.
> `tts-test "a line"` hears the difference immediately. See
> `config/config.yaml`'s `tts:` section for the full set of options.

---

## 3. Fill in `.env`

The installer copied `.env.example` to `.env`. Open it and paste in everything
from step 1:

```bash
nano .env
```

Then regenerate the liquidsoap script so the caster.fm password lands in it:

```bash
.venv/bin/python main.py stream-config
```

That prints `liquidsoap --check: OK` if the script is valid. **If it reports a
syntax error, stop here and fix it** — the message names the line.

---

## 4. Check everything

```bash
.venv/bin/python main.py doctor
```

Every line should be `OK`, with warnings only for things you deliberately
skipped. It makes a real Last.fm call and a real LLM call, so this catches a
bad key immediately rather than at 3am.

---

## 5. Fill the empty library

The library starts empty, so the station has nothing to play yet. Seed artists
live in `config/config.yaml` under `discovery.seed_artists` — **edit those to
artists you actually like first**, because everything the station discovers
grows outward from them.

```bash
nano config/config.yaml     # discovery.seed_artists
.venv/bin/python main.py bootstrap --count 40
```

This downloads one track at a time (deliberately — two cores). Expect roughly
20–40 minutes for 40 tracks on a Celeron with a spinning disk. Rejected
candidates are logged with the reason; `main.py log` shows them.

Check it landed:

```bash
.venv/bin/python main.py status
ls library/ | head
```

---

## 6. Make it headless

```bash
bash scripts/headless-setup.sh
```

Lid close stops mattering, sleep is masked, the console stops blanking. Read
the two manual steps it prints at the end (BIOS auto-power-on, and the desktop
power settings if you run the Mint desktop rather than a console).

Use the **wired ethernet**, not the old 802.11n wifi. Confirm which is in use:

```bash
ip route get 1.1.1.1
```

---

## 7. Start the services

```bash
bash scripts/install-services.sh
```

Three services, deliberately independent:

| Service | What it runs | If it dies |
|---------|--------------|------------|
| `ai-radio-stream` | liquidsoap → caster.fm | Restarts in 10s. This is the one that matters. |
| `ai-radio-brain` | planner, TTS, downloads, queue feeder | Restarts in 15s. The stream doesn't notice. |
| `ai-radio-bot` | Discord | Restarts in 20s. Nothing else notices. |

Watch it come up:

```bash
journalctl -u ai-radio-stream -f
```

You want to see `connected to caster.fm`. Then open your caster.fm listen link
and confirm audio.

The brain needs a minute or two on first run: it plans a block and renders the
patter before anything AI-programmed reaches the stream. Until then the safety
playlist covers — which is exactly the behaviour you want.

```bash
journalctl -u ai-radio-brain -f
```

---

## 8. Prove it never goes silent

This is the one test worth doing by hand, because "never silent" is the whole
point of the architecture.

```bash
# 1. Confirm the stream is playing AI-programmed audio.
.venv/bin/python main.py status

# 2. Kill the brain outright.
sudo systemctl stop ai-radio-brain

# 3. Drain everything the brain had queued, so only the safety tier is left.
.venv/bin/python - <<'PY'
from airadio.config import get_config
from airadio.db import Database
db = Database(get_config().db_path)
print("dropped:", db.execute("DELETE FROM queue_items WHERE status='ready'"))
PY
```

Keep listening. Once liquidsoap has played out the handful of items already
pushed into it (a few minutes), the safety playlist takes over and the music
continues with no gap. Nothing in the logs should show a disconnect.

Then bring it back:

```bash
sudo systemctl start ai-radio-brain
```

Within a couple of minutes the brain plans a fresh block and normal programming
resumes on the next track boundary.

Worth also testing: `sudo systemctl stop ai-radio-stream` then start it again,
and confirm liquidsoap reconnects to caster.fm on its own.

---

## 9. The second host

Upgrading an existing install needs one extra voice model. The installer
fetches it for new installs; on a laptop that is already running:

```bash
cd ~/radioSjors
V=voices          # or wherever your first voice lives
B=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium
curl -fL -o $V/en_GB-jenny_dioco-medium.onnx      $B/en_GB-jenny_dioco-medium.onnx
curl -fL -o $V/en_GB-jenny_dioco-medium.onnx.json $B/en_GB-jenny_dioco-medium.onnx.json
```

It has to sit **next to** the voice already named in `tts.voice_model`, which
is how the co-host is found without spelling out a second path. Then check it:

```bash
.venv/bin/python main.py doctor            # look for "two-host segments"
.venv/bin/python main.py tts-test --duet "Ray goes first|and Nina answers"
```

If the file is missing, nothing breaks — the station simply runs with one host
and never writes a conversation. Rename the hosts, change their characters or
drop the second one entirely under `dj.hosts` in `config/config.local.yaml`.

---

## 10. The website (optional)

A Next.js site under `site/`, deployed on Vercel, showing now playing, the
queue, what just went out and a listen button. It is password-gated, because
the station is private on purpose.

The laptop cannot be reached from the internet, so it pushes instead: a small
JSON document into a secret GitHub Gist every twenty seconds.

**On the laptop**, make a GitHub token at
<https://github.com/settings/tokens> → *Generate new token (classic)*, ticking
**only** the `gist` scope. Put it in `.env` as `GITHUB_TOKEN`, then:

```bash
.venv/bin/python main.py site-init
```

It creates the gist and prints the id. Put that in `.env` as `SITE_GIST_ID`,
turn publishing on in `config/config.local.yaml`:

```yaml
site:
  enabled: true
```

and restart: `sudo systemctl restart ai-radio-brain`. Check it with
`.venv/bin/python main.py site-preview`, which prints exactly what the site
will be told.

**On Vercel**, import this repo and set **Root Directory** to `site`. Then add
these environment variables and redeploy:

| Name | Value |
|------|-------|
| `GIST_ID` | what `site-init` printed |
| `SITE_PASSWORD` | whatever you want to type to get in |
| `STREAM_URL` | the direct listen link from caster.fm |

If `SITE_PASSWORD` is unset, nobody gets in at all — including you. That is on
purpose: a private station should not fall open because a variable was
forgotten.

---

## Tuning it afterwards

Everything lives in `config/config.yaml`, with machine-specific overrides in
`config/config.local.yaml` (which wins). Restart the brain after editing:
`sudo systemctl restart ai-radio-brain`.

Worth knowing about:

| Setting | Does what |
|---------|-----------|
| `dj.persona` | The station's whole personality. Rewrite it freely — it goes into every prompt verbatim. |
| `dj.hosts` | The two hosts: name, character, and which voice model each uses. Delete the second entry to go back to one voice; everything else adapts. |
| `dj.patter_every_n_tracks` | `1` for a chatty station, `3`–`4` for mostly music. |
| `dj.segments` | How much extra talk an hour gets. Each value is a `[low, high]` range rolled fresh every hour. `[0, 0]` switches something off; `[2, 2]` pins it. |
| `dj.max_segment_words` | The ceiling on a long item — a weather moment, a music note, a whole conversation. Raise it if you want them chattier. |
| `dj.max_patter_words` | The ceiling on an ordinary link (back-announce/next-up), separate from `max_segment_words` above. Raise it if the short links between songs feel clipped. |
| `dj.turn_gap_seconds` | Silence inserted between one host finishing and the other starting in a conversation. Too low and two different voices run together like one person; too high reads as dead air instead of a real back-and-forth — tune by ear with `tts-test --duet "Line one|Line two"`. |
| `dj.hosts[].noise_scale` / `noise_w` | Piper's own expressiveness knobs, per host. Higher reads more alive; push either far past ~1.0 and it starts sounding rough instead. There is a real ceiling on how expressive a small offline model gets regardless — this helps, it does not transform it. `tts-test --host Nina "line"` to hear a change without waiting for a whole hour. |
| `tts.engine: edge_tts` + `tts.edge_voice` / `dj.hosts[].edge_voice` | Swap Piper for Microsoft Edge's free neural voices — real prosody, still no bill. Unofficial endpoint, so `voice_model` stays set on every host as the automatic per-line fallback if edge-tts is offline or blocked; `doctor` says which one is actually in use. |
| `weather.*` | Where the weather comes from. Change `place_name`, `latitude` and `longitude` if you move. `enabled: false` and they never mention it. |
| `planner.default_genres` | Genre words per mood_map slot **name** (`{"evening": [jazz]}`), narrowing an hour's candidates to tracks tagged anything close to them before energy is even considered — the reason a rock hour stops pulling in jazz or Christmas music at the same tempo. Loose substring matching, e.g. `rock` also catches `alternative rock`. Skipped automatically if too few of the library are tagged close enough yet (watch for "widening past genre" in the brain log). Keyed by name rather than living on each mood_map entry so it survives a `config.local.yaml` that sets its own `planner.mood_map` — that replaces the whole list, but this dict still merges in by name. An entry can still set its own `genres:` to override this for just that slot. |
| `planner.mood_map` | Time-of-day → mood and energy band. Hour ranges must cover 0–23. Replacing this list in `config.local.yaml` replaces it wholesale — `default_genres` above is what keeps genre filtering working across that override. |
| `planner.buffer_minutes` | How far ahead to work. Lower = the station reacts faster; higher = more slack when things fail. |
| `llm.gemini_model` / `llm.groq_model` | Which model each provider uses. Both retire models regularly — the provider's own `/models` endpoint is the source of truth. |
| `llm.max_tokens` | Completion budget per job. Reasoning models spend part of it thinking, so an empty reply means this is too low, not that the model had nothing to say. |
| `discovery.downloads_per_hour` | Library growth rate. `0`–`3` is plenty. |
| `discovery.library_target` | Stop growing at this many tracks. `0` for no cap. |
| `downloader.reject_title_patterns` | The junk filter. Whole-word matched, so `live` won't reject *Livewire*. |
| `stream.liquidsoap_queue_depth` | How many items sit inside liquidsoap. Lower = mood shifts apply sooner; higher = more slack if the brain stalls. |
| `bot.min_tracks_for_request` | Naming an artist in chat (a request or a mood) tops the library up to this many of their tracks on the spot if it's short. `0` disables the on-demand fetch entirely. |
| `bot.backfill_budget_seconds` | Ceiling on how long that on-demand fetch is allowed to run before giving up with whatever it got. It runs inside the chat reply, so keep this well under a few minutes. |

New downloads have their loudness evened out automatically (a raw YouTube rip
can be mastered anywhere from whisper-quiet to brickwalled). Tracks already in
the library from before this existed are untouched until you run the one-time
backfill:

```bash
.venv/bin/python main.py normalize-library
```

Safe to interrupt and re-run — it only processes tracks it hasn't already
normalised, one ffmpeg pass each.

---

## Day-to-day

```bash
systemctl status ai-radio-stream ai-radio-brain ai-radio-bot
journalctl -u ai-radio-brain -f              # what it's planning
journalctl -u ai-radio-brain --since "1 hour ago" | grep -i reject
.venv/bin/python main.py status              # buffer, mood, now playing
.venv/bin/python main.py log --count 40      # download decisions and rejections
```

After updating yt-dlp (worth doing when downloads start failing — YouTube
changes things):

```bash
.venv/bin/pip install -U yt-dlp
sudo systemctl restart ai-radio-brain
```

---

## Troubleshooting

**The mount is up but silent.** The library is empty, or every file failed to
read. `ls library/` and `main.py status`. If the library really is empty,
liquidsoap's `mksafe` is streaming silence on purpose — that's correct
behaviour, it's holding the mount. Run `bootstrap`.

**Stream won't connect to caster.fm.** `journalctl -u ai-radio-stream -n 50`.
`Connection refused` means nothing is listening: the server is stopped in the
caster.fm dashboard (start it), or the port is wrong. An authentication error
instead means a wrong password, a wrong mount spelling (leading slash!), or
the mount is already in use by an old source, which you can kick from the
dashboard. After changing `.env` you must re-run `main.py stream-config` and
restart the service -- the password lives in the generated `.liq`.

To see just the connection state in a noisy log:

```bash
journalctl -u ai-radio-stream --no-pager | grep -iE "caster|connect" | tail -20
```

**No patter, only music.** TTS isn't working. `main.py doctor` names the reason,
and `main.py tts-test "hello"` tries a single render. The station is *designed*
to keep going in this state, so it won't shout about it.

**Requests don't do anything.** Check `ai-radio-bot` is running, and that
MESSAGE CONTENT INTENT is enabled in the Discord Developer Portal. Without it
the bot receives your messages with the text stripped out.

**Downloads all get rejected.** `main.py log` shows the reason per candidate.
If it's `duration ... off the expected`, Last.fm's duration for that track is
wrong; loosen `downloader.duration_tolerance`. If it's everything at once,
update yt-dlp.

**The LLM stopped answering.** Free tiers rate-limit. With both Gemini and Groq
keys set it fails over automatically, and a rate-limited provider sits out for
5 minutes. If both are out, the deterministic fallback planner takes over and
the station keeps sounding like a station — you'll see
`source=fallback` in the brain log.

---

## A note on the caster.fm tier

Keeping a mount alive 24/7 with no source hiccup ("disable idle timeout") is a
Cloud Plus feature on caster.fm. On the free tier the mount may be dropped if
the source stutters. Liquidsoap reconnects automatically either way, but a
dropped mount means a gap for the listener. Worth confirming which tier you're
on before blaming the software for a dropout.
