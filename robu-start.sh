#!/bin/bash
# Robu one-word launcher.
# - Pulls the latest code from GitHub (both repos)
# - Starts the data server (8000) and the app (3000)
# - Opens the app in your browser
# - Auto-saves & pushes your changes to GitHub (every minute + when you stop)
# Lives inside the repo so it travels via git. Uses relative paths so it works
# on any laptop as long as the two repo folders sit side by side.

HERE="$(cd "$(dirname "$0")" && pwd)"          # .../robu-data-server
DATA="$HERE"
WEB="$(cd "$HERE/../robu-valuation-next" && pwd)"
PUSH_EVERY=60                                   # seconds between auto-pushes

say(){ printf "\n\033[1;36m> %s\033[0m\n" "$1"; }

sync_repo(){  # $1 = repo path
  if [ -n "$(git -C "$1" status --porcelain)" ]; then
    git -C "$1" add -A
    git -C "$1" commit -m "auto-save $(date '+%Y-%m-%d %H:%M')" >/dev/null 2>&1
    git -C "$1" push >/dev/null 2>&1 && printf "  synced %s @ %s\n" "$(basename "$1")" "$(date '+%H:%M')"
  fi
}

cleanup(){
  say "Saving your work to GitHub before stopping..."
  sync_repo "$DATA"; sync_repo "$WEB"
  kill $DPID $WPID 2>/dev/null
  say "Stopped. Everything saved. See you next time."
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
( cd "$DATA" && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 ) >"$HERE/.data.log" 2>&1 &
DPID=$!

say "Starting the app (port 3000)..."
cd "$WEB"
[ -d node_modules ] || npm install
( NODE_OPTIONS=--max-old-space-size=2048 npm run dev ) >"$HERE/.web.log" 2>&1 &
WPID=$!

sleep 6
open http://localhost:3000
say "Robu is running: http://localhost:3000"
echo "   Auto-sync is ON. Your changes save to GitHub automatically."
echo "   When you are done, click this window and press Ctrl + C (it saves + stops)."

while true; do
  sleep "$PUSH_EVERY"
  sync_repo "$DATA"; sync_repo "$WEB"
done
