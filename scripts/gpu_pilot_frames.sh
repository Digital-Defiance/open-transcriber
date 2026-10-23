#!/usr/bin/env bash
# runner: gpu hours=6
# Sung-data frame-model GPU pilot (one detached, resumable, battery-aware, window-capped run).
#
# Chains, ONE GPU job at a time, in a single detached script under caffeinate -i:
#   1. --prep-dev : Demucs vocal stems + Basic Pitch posteriors for the 42 real-dev items (one-time, idempotent).
#   2. a small sung batch + one FrameModel training run: SoulX sings each sampler score, mix over the band,
#      Demucs vocal, BP posteriors + log-mel, rasterise exact labels, train (run_frames.py --songs N --epochs E).
#   3. the fair cross-fit real-dev score (model row + its shift, same scorer as the BP baseline).
#
# Must-haves honoured: resumes from cached stems/posteriors/dataset; pauses while on battery and resumes on AC;
# caps itself to the granted window (reads out/runner/window.json "until"); checkpoints the dataset as it goes.
# Env is the pinned torch env for the frame model; copyist build / sing / Demucs / Basic Pitch shell to the main .venv.

set -uo pipefail

ROOT="/Volumes/Code/copyist/.worktrees/sung-transcriber"
MAIN="/Volumes/Code/copyist"
SUB="$ROOT/open-transcriber"
EARPY="$MAIN/.cache/ear/env/bin/python"
VENVPY="$MAIN/.venv/bin/python"
OUTJUDGE="$MAIN/out/judge"
LOG="$OUTJUDGE/frames-run/gpu_pilot.log"
WINDOW="$MAIN/out/runner/window.json"

SONGS="${PILOT_SONGS:-200}"
EPOCHS="${PILOT_EPOCHS:-20}"

export COPYIST_VENV="$VENVPY"
export COPYIST_SOULX_DIR="${COPYIST_SOULX_DIR:-$MAIN/.cache/SoulX-Singer}"
export COPYIST_SOULX_PYTHON="${COPYIST_SOULX_PYTHON:-$MAIN/.cache/SoulX-Singer/.venv/bin/python}"
export COPYIST_SOULX_DEVICE="${COPYIST_SOULX_DEVICE:-auto}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p "$OUTJUDGE/frames-run"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

window_deadline_epoch() {
  # Reads the ISO "until" from window.json; echoes epoch seconds, or empty if none.
  [ -f "$WINDOW" ] || { echo ""; return; }
  "$EARPY" - "$WINDOW" <<'PY' 2>/dev/null
import json,sys,datetime
try:
    u=json.load(open(sys.argv[1]))["until"]
    print(int(datetime.datetime.fromisoformat(u).timestamp()))
except Exception:
    print("")
PY
}

DEADLINE="$(window_deadline_epoch)"
if [ -n "$DEADLINE" ]; then
  log "window deadline (epoch): $DEADLINE ($(date -r "$DEADLINE" 2>/dev/null))"
else
  log "no window.json deadline found; running without an explicit cap (owner-granted window assumed)"
fi

within_window() {
  [ -z "$DEADLINE" ] && return 0
  [ "$(date +%s)" -lt "$DEADLINE" ] && return 0
  return 1
}

wait_for_ac() {
  while :; do
    if pmset -g batt | grep -q "AC Power"; then return 0; fi
    log "on battery — pausing 60 s, will resume on AC"
    sleep 60
    within_window || { log "window ended while on battery — stopping"; return 1; }
  done
}

guard() {
  within_window || { log "window cap reached — checkpointing and stopping before next stage"; exit 0; }
  wait_for_ac || exit 0
}

log "=== GPU pilot start: songs=$SONGS epochs=$EPOCHS ==="
log "soulx_dir=$COPYIST_SOULX_DIR soulx_py=$COPYIST_SOULX_PYTHON device=$COPYIST_SOULX_DEVICE"

# Stage 1: dev stems + BP posteriors (idempotent, resumable).
guard
log "Stage 1/3: --prep-dev (Demucs vocal stems + BP posteriors for 42 real-dev items)"
cd "$SUB" || exit 1
"$EARPY" scripts/run_frames.py --prep-dev >>"$LOG" 2>&1
log "Stage 1 exit=$?"

# Stage 2+3: sung batch + train + fair cross-fit real-dev eval.
guard
log "Stage 2+3: sung batch ($SONGS songs), train ($EPOCHS epochs), fair cross-fit real-dev eval"
cd "$SUB" || exit 1
"$EARPY" scripts/run_frames.py --songs "$SONGS" --epochs "$EPOCHS" >>"$LOG" 2>&1
RC=$?
log "Stage 2+3 exit=$RC"

if [ "$RC" -eq 0 ] && [ -f "$OUTJUDGE/frames.json" ]; then
  log "=== GPU pilot DONE — results ==="
  cat "$OUTJUDGE/frames.md" | tee -a "$LOG"
else
  log "=== GPU pilot ended rc=$RC (frames.json present: $([ -f "$OUTJUDGE/frames.json" ] && echo yes || echo no)) ==="
fi
