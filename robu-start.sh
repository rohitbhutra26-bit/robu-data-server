#!/bin/bash
# Robu one-word launcher.
# - Pulls the latest code from GitHub (both repos)
# - Starts the data server (8000) and the app (3000)
# - Opens the app in your browser
# - Saves your work locally as you go, and pushes to GitHub ONCE when you stop
#   (Ctrl+C). Pushing only on stop means the live Railway site updates cleanly,
#   never with half-finished, broken-build states.
# Lives inside the repo so it travels via git. Uses relative paths so it works
# on any laptop as long as the two repo folders sit side by side.

HERE="$(cd "$(dirname "$0")" && pwd)"          # .../robu-data-server
DATA="$HERE"
WEB="$(cd "$HERE/../robu-valuation-next" && pwd)"
SAVE_EVERY=120                                  # seconds between local safety-saves
DATA_LOG="/tmp/robu-data.log"                   # logs kept OUT of the repo
WEB_LOG="/tmp/robu-web.log"

say(){ printf "\n\033[1;36m> %s\033[0m\n" "$1"; }

# Local-only commit (no push) — keeps your work safe without touching the live site.
save_local(){  # $1 = repo path
  if [ -n "$(git -C "$1" status --porcelain)" ]; then
    git -C "$1" add -A
    git -C "$1" commit -m "wip $(date '+%Y-%m-%d %H:%M')" >/dev/null 2>&1 \
      && printf "  saved %s @ %s\n" "$(basename "$1")" "$(date '+%H:%M')"
  fi
}

# Commit anything left, then push both repos — this is what deploys to Railway.
push_repo(){  # $1 = repo path
  git -C "$1" add -A
  git -C "$1" commit -m "session $(date '+%Y-%m-%d %H:%M')" >/dev/null 2>&1
  git -C "$1" push >/dev/null 2>&1 && printf "  pushed %s\n" "$(basename "$1")"
}

cleanup(){
  say "Pushing your work to GitHub (this updates the live site)..."
  push_repo "$DATA"; push_repo "$WEB"
  kill $DPID $WPID 2>/dev/null
  say "Stopped. Everything saved and pushed. See you next time."
  exit 0
}
trap cleanup INT TERM

say "Getting the latest from GitHub..."
git -C "$DATA" pull --rebase --autostash 2>&1 | tail -1
git -C "$WEB"  pull --rebase --autostash 2>&1 | tail -1

say "Clearing any old servers..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

say "Starting data server (port 8000)..."
( cd "$DATA" && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 ) >"$DATA_LOG" 2>&1 &
DPID=$!

say "Starting the app (port 3000)..."
cd "$WEB"
# Always install (fast when nothing changed) so new dependencies pulled from
# git are picked up automatically on either laptop.
npm install --no-audit --no-fund --silent
( NODE_OPTIONS=--max-old-space-size=2048 npm run dev ) >"$WEB_LOG" 2>&1 &
WPID=$!

sleep 6
open http://localhost:3000
say "Robu is running: http://localhost:3000"
echo "   Your work auto-saves locally as you go."
echo "   It pushes to GitHub + updates the LIVE site only when you press Ctrl + C."

while true; do
  sleep "$SAVE_EVERY"
  save_local "$DATA"; save_local "$WEB"
done
