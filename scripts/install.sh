#!/usr/bin/env bash
# ===========================================================================
# AI Radio installer for Linux Mint 22.2 / Ubuntu 24.04.
#
#   bash scripts/install.sh
#
# Installs system packages, a Python venv, the Piper voice, and generates the
# liquidsoap script. Safe to re-run.
# ===========================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_DIR="${PROJECT_DIR}/voices"
# The station has two hosts and they need to sound different, so two voices
# are fetched. Both land in the same directory, which is how the co-host is
# found without anyone writing out a second absolute path.
VOICE_NAME="en_US-amy-medium"
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"
COHOST_NAME="en_GB-jenny_dioco-medium"
COHOST_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -ne 0 ] || die "run this as your normal user, not root (it uses sudo where needed)"

say "Installing system packages"
sudo apt-get update
sudo apt-get install -y \
  liquidsoap \
  ffmpeg \
  python3-venv python3-pip \
  curl \
  sqlite3

say "Checking liquidsoap can encode mp3"
if ! liquidsoap --list-plugins 2>/dev/null | grep -qi 'mp3\|lame'; then
  warn "liquidsoap may not have an mp3 encoder; if the stream fails to start, install"
  warn "the full build:  sudo apt-get install liquidsoap-plugin-lame"
fi

say "Creating the Python virtual environment"
python3 -m venv "${PROJECT_DIR}/.venv"
"${PROJECT_DIR}/.venv/bin/pip" install --upgrade pip
"${PROJECT_DIR}/.venv/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"

say "Installing Piper (text to speech)"
# The pip package brings its own CLI. If onnxruntime will not run on this CPU
# (Bay Trail has no AVX) the doctor command will say so and you can switch
# config/config.yaml to  tts.engine: espeak  as a fallback.
if "${PROJECT_DIR}/.venv/bin/pip" install piper-tts; then
  PIPER_OK=1
else
  PIPER_OK=0
  warn "piper-tts failed to install; falling back to espeak-ng"
  sudo apt-get install -y espeak-ng
fi

say "Downloading the voices"
mkdir -p "${VOICE_DIR}"
fetch_voice() {
  local name="$1" base="$2" suffix target
  for suffix in ".onnx" ".onnx.json"; do
    target="${VOICE_DIR}/${name}${suffix}"
    if [ -s "${target}" ]; then
      echo "  already have $(basename "${target}")"
    else
      curl -fL --retry 3 -o "${target}" "${base}/${name}${suffix}"
    fi
  done
}
fetch_voice "${VOICE_NAME}" "${VOICE_BASE}"
# The co-host. If this one fails the station still works -- it just runs with
# a single voice and never writes a conversation.
if ! fetch_voice "${COHOST_NAME}" "${COHOST_BASE}"; then
  warn "could not download the co-host voice (${COHOST_NAME});"
  warn "the station will run with one host until it is there"
fi

say "Writing local configuration overrides"
LOCAL_CONFIG="${PROJECT_DIR}/config/config.local.yaml"
if [ -f "${LOCAL_CONFIG}" ]; then
  echo "  ${LOCAL_CONFIG} already exists, leaving it alone"
else
  if [ "${PIPER_OK}" -eq 1 ]; then
    ENGINE="piper_python"
  else
    ENGINE="espeak"
  fi
  cat > "${LOCAL_CONFIG}" <<EOF
# Machine-specific overrides. This file wins over config/config.yaml and is
# not tracked, so edit it freely.
tts:
  engine: ${ENGINE}
  voice_model: ${VOICE_DIR}/${VOICE_NAME}.onnx
EOF
  echo "  wrote ${LOCAL_CONFIG}"
fi

if [ ! -f "${PROJECT_DIR}/.env" ]; then
  cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
  chmod 600 "${PROJECT_DIR}/.env"
  warn "created .env from the example - fill in your keys before starting"
fi

say "Initialising directories and database"
"${PROJECT_DIR}/.venv/bin/python" "${PROJECT_DIR}/main.py" init

say "Generating the liquidsoap script"
"${PROJECT_DIR}/.venv/bin/python" "${PROJECT_DIR}/main.py" stream-config

cat <<EOF

---------------------------------------------------------------------------
Installed.  Next:

  1. Edit .env and put in your caster.fm details, Last.fm key, an LLM key
     (Gemini and/or Groq) and your Discord bot token.

  2. Re-run the config generator so the password lands in the .liq script:
       .venv/bin/python main.py stream-config

  3. Check everything:
       .venv/bin/python main.py doctor

  4. Fill the empty library (this takes a while, one download at a time):
       .venv/bin/python main.py bootstrap --count 40

  5. Install the services:
       bash scripts/install-services.sh

See SETUP.md for the full walkthrough.
---------------------------------------------------------------------------
EOF
