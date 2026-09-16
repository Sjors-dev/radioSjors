#!/usr/bin/env bash
# ===========================================================================
# Turn the laptop into a headless 24/7 box: no sleep on lid close, no display
# blanking, no automatic suspend.
#
#   bash scripts/headless-setup.sh
#
# The battery being dead is fine - it runs on mains. Use wired ethernet.
# ===========================================================================
set -euo pipefail

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

say "Ignoring the lid switch"
sudo mkdir -p /etc/systemd/logind.conf.d
sudo tee /etc/systemd/logind.conf.d/99-ai-radio.conf > /dev/null <<'EOF'
[Login]
HandleLidSwitch=ignore
HandleLidSwitchDocked=ignore
HandleLidSwitchExternalPower=ignore
EOF
sudo systemctl restart systemd-logind || true

say "Masking sleep, suspend and hibernate targets"
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

say "Disabling console blanking on the local TTY"
sudo tee /etc/systemd/system/disable-console-blank.service > /dev/null <<'EOF'
[Unit]
Description=Disable console blanking
After=getty.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c '/usr/bin/setterm --blank 0 --powerdown 0 < /dev/tty1 || true'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now disable-console-blank.service || true

say "Current power settings"
systemctl is-enabled sleep.target 2>/dev/null || echo "  sleep.target: masked (good)"

cat <<'EOF'

Done. Two things to do by hand:

  1. BIOS: enable "Restore on AC power loss" / "Power On by AC" if the ASUS
     has it, so the laptop comes back after a power cut.

  2. If you use the Mint desktop rather than a bare console, also set
     Power Management -> "When the lid is closed: Do nothing" and turn off
     screen blanking, because the desktop session overrides logind.

Check the ethernet link is the one in use:
  ip route get 1.1.1.1
EOF
