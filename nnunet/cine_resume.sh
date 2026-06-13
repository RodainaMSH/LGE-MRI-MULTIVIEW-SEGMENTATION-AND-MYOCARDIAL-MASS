#!/usr/bin/env bash
# Resumable, idempotent Cine training — survives reboots/power loss.
# Safe to run ANYTIME (Windows logon task, .bashrc hook, or by hand): it
#   (a) is a singleton (flock) so it never double-runs,
#   (b) EXITS immediately if any nnUNetv2_train is already running (no GPU conflict with the live run),
#   (c) SKIPS folds that already have checkpoint_final.pth,
#   (d) RESUMES an interrupted fold with --c (from checkpoint_latest), else trains fresh.
# => after any crash/outage, just (re)running this picks up exactly where it left off.
ROOT=/home/youssef/projects/research/CMR-MULTI
cd "$ROOT" || exit 1
exec 9>/tmp/cine_resume.lock; flock -n 9 || { echo "[cine_resume] already running"; exit 0; }
# don't fight a live training run (the original orchestrator, or another resume)
if pgrep -f "nnUNetv2_train 10[012] " >/dev/null 2>&1; then echo "[cine_resume] a training is already active — nothing to do"; exit 0; fi
source nnunet/env.sh
TR=nnUNetTrainer_100epochs
declare -A VN=( [100]=SAX [101]=2CH [102]=4CH )
LOG=nnunet/results/logs/cine_resume.log
echo "[$(date '+%F %T')] cine_resume START" >> "$LOG"
for d in 100 101 102; do
  RUN=$(ls -d nnunet/nnUNet_results/Dataset${d}_*/${TR}__nnUNetPlans__2d 2>/dev/null | head -1)
  for f in 0 1 2 3 4; do
    fdir="$RUN/fold_$f"
    if [ -f "$fdir/checkpoint_final.pth" ]; then echo "[$(date '+%T')] skip ${VN[$d]} f$f (done)" >> "$LOG"; continue; fi
    C=""; [ -f "$fdir/checkpoint_latest.pth" ] && C="--c"
    echo "[$(date '+%T')] train ${VN[$d]} f$f $C" >> "$LOG"
    nnUNetv2_train $d 2d $f -tr $TR $C >> nnunet/results/logs/cine_d${d}_f${f}.log 2>&1
    echo "[$(date '+%T')]   ${VN[$d]} f$f exit $?" >> "$LOG"
  done
done
echo "[$(date '+%F %T')] CINE_TRAIN_ALL_DONE" >> "$LOG"
touch nnunet/results/logs/CINE_TRAIN_ALL_DONE
