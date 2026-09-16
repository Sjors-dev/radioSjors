#!/usr/bin/env bash
# ===========================================================================
# Install the three systemd services. Run after scripts/install.sh and after
# .env is filled in.
#
#   bash scripts/install-services.sh
# ===========================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$USER}"
UNITS=(ai-radio-stream ai-radio-brain ai-radio-bot)

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

[ -f "${PROJECT_DIR}/stream/radio.generated.liq" ] \
  || die "stream/radio.generated.liq is missing - run: .venv/bin/python main.py stream-config"
[ -x "${PROJECT_DIR}/.venv/bin/python" ] \
  || die ".venv is missing - run: bash scripts/install.sh"

say "Installing units for user ${RUN_USER} from ${PROJECT_DIR}"
for unit in "${UNITS[@]}"; do
  sed -e "s|@PROJECT_DIR@|${PROJECT_DIR}|g" \
      -e "s|@USER@|${RUN_USER}|g" \
      "${PROJECT_DIR}/services/${unit}.service" \
    | sudo tee "/etc/systemd/system/${unit}.service" > /dev/null
  echo "  /etc/systemd/system/${unit}.service"
done

sudo systemctl daemon-reload

say "Enabling and starting"
# The stream goes first: it must be up and holding the mount before anything
# else. The brain and bot can come and go without touching it.
sudo systemctl enable --now ai-radio-stream.service
sleep 3
sudo systemctl enable --now ai-radio-brain.service

if grep -qE '^DISCORD_TOKEN=.+' "${PROJECT_DIR}/.env"; then
  sudo systemctl enable --now ai-radio-bot.service
else
  echo "  DISCORD_TOKEN is empty, leaving the bot disabled for now."
  echo "  Enable it later with: sudo systemctl enable --now ai-radio-bot"
fi

say "Status"
for unit in "${UNITS[@]}"; do
  printf '  %-22s %s\n' "${unit}" "$(systemctl is-active "${unit}" 2>/dev/null || true)"
done

cat <<EOF

Follow the logs with:
  journalctl -u ai-radio-stream -f
  journalctl -u ai-radio-brain -f
  journalctl -u ai-radio-bot -f
EOF
