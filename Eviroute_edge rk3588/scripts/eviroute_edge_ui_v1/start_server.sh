#!/usr/bin/env bash
set -euo pipefail
cd /root/EviRoute_RK3588
mkdir -p logs
exec python3 -u scripts/eviroute_edge_ui_v1/server.py \
  --project-root /root/EviRoute_RK3588 \
  --data-root /root/EviRoute_RK3588/eviroute_edge_data \
  --host 127.0.0.1 --port 8765
