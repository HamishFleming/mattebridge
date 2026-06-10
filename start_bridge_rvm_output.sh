#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
  PYTHON="$SCRIPT_DIR/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "python3 not found on PATH" >&2
  exit 1
fi

if [[ ! -e /dev/video10 ]]; then
  echo "/dev/video10 is missing." >&2
  echo "Load v4l2loopback first:" >&2
  echo "sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=rvm exclusive_caps=1" >&2
  exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/bridge_rvm_output.py" "$@"
