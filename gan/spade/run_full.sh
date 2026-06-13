#!/usr/bin/env bash
# SPADE GAN-aug v2 — full hands-off pipeline (launched AFTER fold-0 gate passes).
# Idempotent: skips a GAN fold whose G_final.pth exists. Mirrors v1's leakage-clean build + the base
# DenseNet-264 recipe EXACTLY (same trainer, same plan transferred from Dataset000) so the ONLY change
# vs the base OOF (scar 0.3925) is the +300/fold SPADE-rendered, on-ring-deformed synthetic scar.
set +e
cd /home/youssef/projects/research/CMR-MULTI
source nnunet/env.sh
export PYTHONPATH=/home/youssef/projects/research/CMR-MULTI
export DN264_CKPT=/home/youssef/projects/research/CMR-MULTI/weights/densenet264/densenet264_in1k_torch.pth
PY=/home/youssef/miniconda3/bin/python
TR=nnUNetTrainerSMP_densenet264
DST=nnunet/nnUNet_raw/Dataset022_LGEspade
RUN=gan/spade/runs
LOG=$RUN/pipeline.log
say(){ echo -e "\n$(date '+%F %T') | $*" | tee -a "$LOG"; }

say "===== SPADE v2 FULL PIPELINE START ====="

# Plain SPADE (label-only) chosen over StyleSPADE: at our N=32/fold the style-conditioned generator
# overfits (recon corr 0.98) and leaks the old scar location as a GHOST (+0.13, 51% of cases) = label noise.
# Plain SPADE renders scar purely from the deformed label -> ghost-free (+0.026) AND higher new-scar
# contrast (+0.349 vs StyleSPADE +0.272, v1 +0.296). Fold-0 already trained at gan/spade/runs/f0_spade.

# ---- 1. per-fold plain-SPADE GANs (fold 0 trained separately; train 1-4 here) ----
for k in 1 2 3 4; do
  if [ -f $RUN/f${k}_spade/G_final.pth ]; then say "GAN fold $k already done — skip"; continue; fi
  say "GAN fold $k train (plain SPADE)"
  $PY -u gan/spade/train_spade.py --views SAX,2CH,4CH --exclude_fold $k --netG spade \
      --niter 150 --niter_decay 50 --bs 5 --out $RUN/f${k}_spade > $RUN/f${k}_spade.log 2>&1
  say "  GAN fold $k done (exit $?)"
done

# ---- 2. generate per-fold synthetic (300/fold, on-ring deformed, leakage-clean) into Dataset022 ----
rm -rf "$DST" nnunet/nnUNet_preprocessed/Dataset022_LGEspade nnunet/nnUNet_results/Dataset022_LGEspade
for k in 0 1 2 3 4; do
  say "gen fold $k -> 300 synthetic (plain SPADE)"
  $PY -u gan/spade/gen_spade.py --gan $RUN/f${k}_spade/G_final.pth --exclude_fold $k --views SAX,2CH,4CH \
      --netG spade --n 300 --prefix SYNf$k --out_img $DST/imagesTr --out_lab $DST/labelsTr >> $RUN/gen.log 2>&1
done
NSYN=$(ls "$DST"/labelsTr/SYNf*.nii.gz 2>/dev/null | wc -l)
say "generated $NSYN synthetic cases"

# ---- 3. build Dataset022 = real + synthetic, plan transferred from Dataset000 (identical recipe) ----
$PY gan/spade/build_dataset022.py --mode assemble >> $RUN/build.log 2>&1
nnUNetv2_extract_fingerprint -d 22 >> $RUN/build.log 2>&1
nnUNetv2_move_plans_between_datasets -s 0 -t 22 -sp nnUNetPlans -tp nnUNetPlans >> $RUN/build.log 2>&1
cp "$DST"/dataset.json nnunet/nnUNet_preprocessed/Dataset022_LGEspade/dataset.json
nnUNetv2_preprocess -d 22 -c 2d -plans_name nnUNetPlans >> $RUN/build.log 2>&1
$PY gan/spade/build_dataset022.py --mode splits >> $RUN/build.log 2>&1
PATCH=$($PY -c "import json;print(json.load(open('nnunet/nnUNet_preprocessed/Dataset022_LGEspade/nnUNetPlans.json'))['configurations']['2d']['patch_size'])" 2>/dev/null)
say "Dataset022 built — $NSYN synth, 2d patch_size=$PATCH"

# ---- 4. DenseNet-264 5-fold on Dataset022 (same trainer/recipe as base) ----
for f in 0 1 2 3 4; do
  say "DenseNet fold $f train (Dataset022)"
  nnUNetv2_train 22 2d $f -tr $TR > nnunet/results/logs/spade_dn264_fold${f}.log 2>&1
  rc=$?; say "  DenseNet fold $f done (exit $rc)"
  if [ $f -eq 0 ] && [ $rc -ne 0 ]; then say "  FOLD-0 FAILED -> abort"; tail -n 15 nnunet/results/logs/spade_dn264_fold0.log | tee -a "$LOG"; exit 1; fi
done

# ---- 5. score OOF vs base (symlink Dataset022 run into Dataset000 tree, as v1 did) ----
SRC=nnunet/nnUNet_results/Dataset022_LGEspade/${TR}__nnUNetPlans__2d
LINK=nnunet/nnUNet_results/Dataset000_LGEgeneralist/spade_dn264__nnUNetPlans__2d
rm -f "$LINK"; ln -s "$(pwd)/$SRC" "$LINK"
say "OOF score (SPADE GAN-aug) [base: scar 0.3925, DSC 0.6903, vPCC 0.650]:"
$PY nnunet/oof_eval_run.py spade_dn264__nnUNetPlans__2d 2>&1 | grep -aE "OOF CV DSC|scar:|REFIT|vPCC|RAE" | tee -a "$LOG"
say "===== SPADE v2 FULL PIPELINE DONE =====" | tee -a "$LOG"
touch $RUN/SPADE_OOF_DONE
