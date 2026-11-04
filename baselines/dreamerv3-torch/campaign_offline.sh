#!/usr/bin/env bash
# campaign_offline.sh — DreamerV3-offline phased campaign (18 runs, 3 seeds × 6 envs)
# Max 4 concurrent dreamer.py runs at all times; nice -19 everything.
# Phase order: all 6 envs at seed 0, then seed 1, then seed 2.
# Each run:
#   1. seed demos into runs/<env>_seed<S>/train_eps/
#   2. setsid nice -19 dreamer.py ... &  (logs to runs/<env>_seed<S>.log)
# Slot-waiter uses pgrep on "dreamer.py --configs uw_defaults" — never matches self.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
MAX_CONCURRENT=4

# Per-env step budgets (base config steps + seeding offset from previous report)
declare -A STEPS=(
    [blocks_rand6]=120780
    [cornergoal]=40171
    [empty5x5]=10060
    [fourrooms]=60478
    [lavawall]=40168
    [unlock]=40236
)

ENVS=(blocks_rand6 cornergoal empty5x5 fourrooms lavawall unlock)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_for_slot() {
    while true; do
        # Count running dreamer.py processes only (won't match this bash script)
        N=$(pgrep -fc "dreamer.py --configs uw_defaults" 2>/dev/null || true)
        if [ "$N" -lt "$MAX_CONCURRENT" ]; then
            return 0
        fi
        log "  $N/$MAX_CONCURRENT slots used — waiting 30s..."
        sleep 30
    done
}

wait_for_phase() {
    # Wait until no dreamer.py processes remain before starting next seed phase
    log "Waiting for current phase to complete..."
    while true; do
        N=$(pgrep -fc "dreamer.py --configs uw_defaults" 2>/dev/null || true)
        if [ "$N" -eq 0 ]; then
            log "Phase complete (0 dreamer.py processes running)."
            return 0
        fi
        log "  $N run(s) still active — checking again in 60s..."
        sleep 60
    done
}

launch_run() {
    local ENV=$1
    local SEED=$2
    local LOGDIR="$REPO/runs/${ENV}_seed${SEED}"
    local LOGFILE="$REPO/runs/${ENV}_seed${SEED}.log"
    local STEPS_ENV="${STEPS[$ENV]}"

    mkdir -p "$LOGDIR"

    log "Seeding demos for $ENV seed=$SEED -> $LOGDIR/train_eps"
    nice -n 19 "$PYTHON" -m envs.uw_seed_demos \
        --name "uw_${ENV}" \
        --outdir "$LOGDIR/train_eps" \
        2>&1 | tee -a "$LOGFILE"

    log "Launching dreamer.py for $ENV seed=$SEED (steps=$STEPS_ENV) -> $LOGFILE"
    setsid nice -n 19 "$PYTHON" \
        "$REPO/dreamer.py" \
        --configs uw_defaults "uw_${ENV}" \
        --logdir "$LOGDIR" \
        --seed "$SEED" \
        --compile False \
        --steps "$STEPS_ENV" \
        --prefill 0 \
        >> "$LOGFILE" 2>&1 &

    log "  Launched PID=$! for $ENV seed=$SEED"
}

# ── Phase 1: seed 0, all 6 envs ─────────────────────────────────────────────
log "=== PHASE 1: seed 0 ==="
for ENV in "${ENVS[@]}"; do
    wait_for_slot
    launch_run "$ENV" 0
done

wait_for_phase

# ── Phase 2: seed 1, all 6 envs ─────────────────────────────────────────────
log "=== PHASE 2: seed 1 ==="
for ENV in "${ENVS[@]}"; do
    wait_for_slot
    launch_run "$ENV" 1
done

wait_for_phase

# ── Phase 3: seed 2, all 6 envs ─────────────────────────────────────────────
log "=== PHASE 3: seed 2 ==="
for ENV in "${ENVS[@]}"; do
    wait_for_slot
    launch_run "$ENV" 2
done

wait_for_phase

log "=== CAMPAIGN COMPLETE: all 18 runs finished ==="
