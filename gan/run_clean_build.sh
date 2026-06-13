#!/bin/bash
# Phase 2-4: after the 5 per-fold GANs finish, quality-gate each, generate ~300/fold leakage-clean synthetic
# (per-fold passing views), assemble Dataset021, and preprocess it with Dataset000's plan (fair comparison).
# Ends by writing BUILD_DONE. Does NOT train — training is launched separately after verification.
cd /home/youssef/projects/research/CMR-MULTI
source nnunet/env.sh >/dev/null 2>&1
LOG=nnunet/decoder_logs
DST=nnunet/nnUNet_raw/Dataset021_LGEganclean
mkdir -p "$LOG"
PIPE="$LOG/ganclean_pipe.log"
echo "[build] $(date '+%F %T') START — waiting for 5 GANs (G_final.pth)" | tee "$PIPE"

for k in 0 1 2 3 4; do
  while [ ! -e gan/runs/allv_f${k}/G_final.pth ]; do sleep 60; done
  echo "[build] $(date '+%T') GAN fold $k ready" | tee -a "$PIPE"
done
echo "[build] $(date '+%T') all 5 GANs done — quality gate + generate" | tee -a "$PIPE"

# clean slate for Dataset021 (raw + preprocessed + results)
rm -rf "$DST" nnunet/nnUNet_preprocessed/Dataset021_LGEganclean nnunet/nnUNet_results/Dataset021_LGEganclean
: > "$LOG/ganclean_gate.log"; : > "$LOG/ganclean_gen.log"; : > "$LOG/ganclean_build.log"

for k in 0 1 2 3 4; do
  PYTHONPATH=. python gan/quality_gate.py --gan gan/runs/allv_f${k}/G_final.pth --views SAX,2CH,4CH --exclude_fold $k >> "$LOG/ganclean_gate.log" 2>&1
  VIEWS=$(grep "GATE_VIEWS=" "$LOG/ganclean_gate.log" | tail -1 | cut -d= -f2)
  [ -z "$VIEWS" ] && VIEWS=SAX
  echo "[build] $(date '+%T') fold $k gate -> views=$VIEWS" | tee -a "$PIPE"
  PYTHONPATH=. python gan/gen_synthetic.py --gan gan/runs/allv_f${k}/G_final.pth --n 300 --views "$VIEWS" \
    --out "$DST" --prefix SYNf${k} --exclude_fold $k >> "$LOG/ganclean_gen.log" 2>&1
done
NSYN=$(ls "$DST"/labelsTr/SYNf*.nii.gz 2>/dev/null | wc -l)
echo "[build] $(date '+%T') generated $NSYN synthetic cases" | tee -a "$PIPE"

# assemble raw, then preprocess Dataset021 with Dataset000's plan (identical patch/spacing/norm)
PYTHONPATH=. python gan/build_clean_dataset.py --mode assemble >> "$LOG/ganclean_build.log" 2>&1
nnUNetv2_extract_fingerprint -d 21 >> "$LOG/ganclean_build.log" 2>&1
nnUNetv2_move_plans_between_datasets -s 0 -t 21 -sp nnUNetPlans -tp nnUNetPlans >> "$LOG/ganclean_build.log" 2>&1
  cp "$DST"/dataset.json nnunet/nnUNet_preprocessed/Dataset021_LGEganclean/dataset.json
nnUNetv2_preprocess -d 21 -c 2d -plans_name nnUNetPlans >> "$LOG/ganclean_build.log" 2>&1
PYTHONPATH=. python gan/build_clean_dataset.py --mode splits >> "$LOG/ganclean_build.log" 2>&1

PATCH=$(python -c "import json;print(json.load(open('nnunet/nnUNet_preprocessed/Dataset021_LGEganclean/nnUNetPlans.json'))['configurations']['2d']['patch_size'])" 2>/dev/null)
echo "[build] $(date '+%T') BUILD_DONE — $NSYN synth, 2d patch_size=$PATCH" | tee -a "$PIPE"
