# The station's front page

A small Next.js site that shows what the radio is playing: now playing, what
is queued, what just went out, and a button to listen.

It is deployed on Vercel from the `site/` folder of this repo. The radio
itself does not run here and never will — this is a window onto it.

## How it gets its data

The radio runs on a laptop behind a home router, so nothing on the internet
can ask it anything. It pushes instead:

```
laptop ──every 20s──▶ secret GitHub Gist ◀──on request── this site ──▶ browser
```

The GitHub token on the laptop only ever holds the `gist` scope, so the worst
thing a leak could do is edit a file that says what song is on.

## Setting it up

**1. Make the gist.** On the laptop, once:

```bash
cd ~/radioSjors && .venv/bin/python main.py site-init
```

It prints a `SITE_GIST_ID`. Put that in the laptop's `.env`, then turn
publishing on in `config/config.local.yaml`:

```yaml
site:
  enabled: true
```

and restart the brain: `sudo systemctl restart ai-radio-brain`.

**2. Deploy.** Import this repo in Vercel and set **Root Directory** to
`site`. Everything else is detected.

**3. Environment variables** (Vercel → Settings → Environment Variables):

| Name               | What it is                                                     |
| ------------------ | -------------------------------------------------------------- |
| `GIST_ID`          | From `site-init`. Required.                                     |
| `SITE_PASSWORD`    | The password for the whole site. Required.                      |
| `STREAM_URL`       | The direct listen link from caster.fm. Gives the nicer player.  |
| `PLAYER_EMBED_URL` | caster.fm's own embed, used only when there is no `STREAM_URL`. |
| `GITHUB_TOKEN`     | Optional. A secret gist reads fine without one.                 |
| `GIST_FILENAME`    | Optional, defaults to `radio.json`.                             |

Then redeploy so the variables are picked up.

## The password

There is one shared password and no accounts. The cookie holds a hash derived
from `SITE_PASSWORD`, so it cannot be forged without knowing it, and nothing
needs a session store.

**If `SITE_PASSWORD` is unset, nobody gets in.** That is deliberate. The
station is private on purpose — it is what keeps music licensing out of the
picture — and a missing environment variable should not quietly open it up to
the whole internet.

## Running it locally

```bash
cd site
npm install
cp .env.example .env.local   # fill in GIST_ID and SITE_PASSWORD
npm run dev
```

## Layout

```
app/
  page.tsx            server component: reads the gist once for first paint
  login/page.tsx      the door
  api/state/route.ts  the same read, for polling, so the token stays server-side
  api/login/route.ts  checks the password, sets the cookie
  globals.css         the whole design, by hand
components/
  Station.tsx         the page itself, polls every 12s
  Player.tsx          audio element and controls
  Stamp.tsx           the rotating thing in the corner
lib/
  station.ts          reads and types the gist payload
  auth.ts             password hashing, shared with proxy.ts
proxy.ts              locks every route except the door
```
