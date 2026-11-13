#!/usr/bin/env bash
# Launch the graphical low-light restoration window.
#
# Primary interface. The CLI (python -m lowlight.cli) is a scripting helper.
# If Tkinter lives in a user-local extraction (no root install), set
# LOWLIGHT_TK_ROOT to that extraction root; otherwise a normal system
# python3 with tkinter works directly.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ -n "${LOWLIGHT_TK_ROOT:-}" && -d "$LOWLIGHT_TK_ROOT" ]]; then
  export PYTHONPATH="$LOWLIGHT_TK_ROOT/usr/lib/python3.11:$LOWLIGHT_TK_ROOT/usr/lib/python3.11/lib-dynload${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$LOWLIGHT_TK_ROOT/usr/lib:${LD_LIBRARY_PATH:-}"
  # Let fontconfig/Tk discover user-local fonts (e.g. WenQuanYi CJK) when the
  # system font directory is not writable.
  if [[ -f "$HOME/.local/fonts.conf" ]]; then
    export FONTCONFIG_FILE="$HOME/.local/fonts.conf"
    mkdir -p /tmp/fontconfig-cache
  fi
fi

export MPLBACKEND="${MPLBACKEND:-TkAgg}"
exec python3 -m lowlight.gui "$@"
