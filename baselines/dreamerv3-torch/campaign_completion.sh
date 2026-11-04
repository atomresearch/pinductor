#!/bin/bash
# Completion pass: finish the truncated seed-1 runs (resume from latest.pt in
# their existing logdirs) and run the full seed-2 phase. 3 concurrent max.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
declare -A STEPS=( [blocks_rand6]=120780 [cornergoal]=40171 [empty5x5]=10060
                   [fourrooms]=60478 [lavawall]=40168 [unlock]=40236 )
RUNS="blocks_rand6:1 lavawall:1 fourrooms:1 blocks_rand6:2 cornergoal:2 empty5x5:2 fourrooms:2 lavawall:2 unlock:2"
log() { echo "[$(date '+%F %T')] $*"; }

for rs in $RUNS; do
  ENV=${rs%:*}; SEED=${rs#*:}
  LOGDIR=$REPO/runs/${ENV}_seed${SEED}
  LOGFILE=$REPO/runs/${ENV}_seed${SEED}.log
  while [ "$(pgrep -fc 'dreamer.py --configs uw_defaults')" -ge 3 ]; do sleep 60; done
  mkdir -p "$LOGDIR"
  if [ ! -d "$LOGDIR/train_eps" ] || [ -z "$(ls -A "$LOGDIR/train_eps" 2>/dev/null)" ]; then
    log "Seeding demos for $ENV seed=$SEED"
    nice -n 19 "$PYTHON" -m envs.uw_seed_demos --name "uw_${ENV}" \
      --outdir "$LOGDIR/train_eps" >> "$LOGFILE" 2>&1
  fi
  log "Launching $ENV seed=$SEED (steps=${STEPS[$ENV]}, resume if checkpoint)"
  setsid nice -n 19 "$PYTHON" "$REPO/dreamer.py" \
    --configs uw_defaults "uw_${ENV}" --logdir "$LOGDIR" --seed "$SEED" \
    --compile False --steps "${STEPS[$ENV]}" --prefill 0 \
    >> "$LOGFILE" 2>&1 &
  sleep 20
done
while pgrep -f 'dreamer.py --configs uw_defaults' > /dev/null; do sleep 120; done
log "COMPLETION PASS DONE"
