#!/bin/bash
# Double-click (or run from VS Code: Terminal > Run Task > Start mirror demo)
cd "$(dirname "$0")"
get() {  # one-time download of a tracking model so the mirror works offline afterwards
  if [ ! -s "$1" ]; then
    echo "Downloading $1 (one time)…"
    curl -fsSL -o "$1" "$2" || rm -f "$1"
  fi
}
get lib/mediapipe/pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
get lib/mediapipe/face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
PORT=8000
while lsof -i :$PORT >/dev/null 2>&1; do PORT=$((PORT+1)); done
URL="http://localhost:$PORT/"
echo "Mirror running at $URL  (close this window or press Ctrl+C to stop)"
( sleep 1; open -a "Google Chrome" "$URL" 2>/dev/null || open "$URL" ) &
exec python3 -m http.server $PORT
