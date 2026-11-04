#!/bin/bash
# Demo-seeded DreamerV3 on POPGym MineSweeper — closes the MineSweeper gap of
# campaign_offline.sh (which covered 6 envs only).
# Same stack and hyperparameters as the other demo-seeded runs (uw_defaults:
# envs 1, batch 16x16, train_ratio 256, prefill 0); per-env step budgets copied
# from the vanilla JAX runs (dreamer_night/ms_*_10k config.yaml).
# Only difference vs vanilla: the replay is pre-seeded with the same demos the
# LLM methods receive (MineSweeper{Easy,Medium,Hard}_random_N50.pkl).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
export UW_REPO="$(cd "$REPO/../.." && pwd)"   # repository root
export LLM_BRIDGE_DIR=/tmp/dummy_bridge
declare -A STEPS=( [ms_easy]=78700 [ms_medium]=169000 [ms_hard]=303000 )
SEED=${1:-0}
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$REPO/campaign_ms_seeded.log"; }

log "=== MS seeded campaign, seed=$SEED ==="
for ENV in ms_easy ms_medium ms_hard; do
    LOGDIR="$REPO/runs/${ENV}_seed${SEED}"
    LOGFILE="$REPO/runs/${ENV}_seed${SEED}.log"
    mkdir -p "$LOGDIR"
    log "Seeding demos for $ENV seed=$SEED -> $LOGDIR/train_eps"
    nice -n 19 "$PYTHON" -m envs.uw_seed_demos --name "uw_${ENV}" \
        --outdir "$LOGDIR/train_eps" 2>&1 | tee -a "$LOGFILE"
    log "Launching dreamer.py for $ENV seed=$SEED (steps=${STEPS[$ENV]})"
    setsid nice -n 19 "$PYTHON" "$REPO/dreamer.py" \
        --configs uw_defaults "uw_${ENV}" \
        --logdir "$LOGDIR" --seed "$SEED" --compile False \
        --steps "${STEPS[$ENV]}" --prefill 0 >> "$LOGFILE" 2>&1 &
    log "  PID=$!"
    sleep 5
done
log "3 runs launched (seed $SEED)"
