#!/usr/bin/env bash
# Run as root over SSH after the graphical desktop user "topeet" has logged in.
set -euo pipefail
GUI_USER="${1:-topeet}"
GUI_UID="$(id -u "$GUI_USER")"
PORT="${EVIR_OUT_PORT:-8765}"
GUI_HOME="$(getent passwd "$GUI_USER" | cut -d: -f6)"
DISPLAY_NUM="${DISPLAY:-:0}"
export GUI_UID GUI_HOME DISPLAY_NUM PORT
if ! pgrep -f "[s]erver.py.*--port ${PORT}" >/dev/null; then
  echo "NOTE: Start server.py first (or confirm local port ${PORT})."
fi
runuser -u "$GUI_USER" -- env \
  DISPLAY="$DISPLAY_NUM" \
  XDG_RUNTIME_DIR="/run/user/$GUI_UID" \
  DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$GUI_UID/bus" \
  XAUTHORITY="$GUI_HOME/.Xauthority" \
  chromium --no-first-run --no-default-browser-check \
  --disable-session-crashed-bubble \
  --user-data-dir="$GUI_HOME/.cache/eviroute-edge-kiosk" \
  --kiosk "http://127.0.0.1:$PORT"
