#!/usr/bin/env bash
#
# start.sh — one command to run the YoloRAG dev stack.
#
#   1. Activates the conda env (default: py312).
#   2. Installs backend deps (pip install -e .) if not already installed.
#   3. Installs frontend deps (npm install) if node_modules is missing.
#   4. Runs the FastAPI backend and the Vite frontend together.
#
# Usage:  ./start.sh
# Stop:   Ctrl+C (stops both processes).
#
# Env overrides:
#   YOLORAG_CONDA_ENV   conda env name         (default: py312)
#   BACKEND_HOST        backend bind host      (default: 127.0.0.1)
#   BACKEND_PORT        backend port           (default: 8000)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA_ENV="${YOLORAG_CONDA_ENV:-py312}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

log() { printf '\033[1;36m▶ %s\033[0m\n' "$*"; }

# --- 1. Activate the conda env ---------------------------------------------
if [ -n "${CONDA_EXE:-}" ] && [ -f "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
else
  for base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/homebrew/Caskroom/miniforge/base; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      source "$base/etc/profile.d/conda.sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "error: could not find conda. Install Miniforge/Miniconda or set CONDA_EXE." >&2
  exit 1
fi

log "Activating conda env: $CONDA_ENV"
conda activate "$CONDA_ENV"

# --- 2. Backend deps (install only if missing) -----------------------------
if ! pip show yolorag >/dev/null 2>&1; then
  log "Installing backend deps (pip install -e .) — first run only"
  pip install -e .
else
  log "Backend deps already installed"
fi

# --- 3. Frontend deps (install only if missing) ----------------------------
if [ ! -d frontend/node_modules ]; then
  log "Installing frontend deps (npm install) — first run only"
  (cd frontend && npm install)
else
  log "Frontend deps already installed"
fi

# --- 4. Run backend + frontend, stop both on exit --------------------------
pids=()
cleanup() {
  trap - EXIT INT TERM
  log "Shutting down…"
  for pid in "${pids[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

log "Backend  → http://$BACKEND_HOST:$BACKEND_PORT"
PYTHONPATH=src uvicorn yolorag.api.app:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
pids+=($!)

log "Frontend → http://127.0.0.1:5173  (Open dataset in the sidebar)"
(cd frontend && npm run dev) &
pids+=($!)

# Wait for either process; Ctrl+C triggers cleanup for both.
wait
