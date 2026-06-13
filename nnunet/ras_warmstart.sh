#!/usr/bin/env bash
# RAS warm-start: ImageNet -> cardiac-LGE generalist -> right-atrium.
# EffNetV2-XL smp.Unet, encoder+decoder loaded from the trained generalist
# (nnUNetTrainerSMP_effv2xl Dataset000 fold_0), only the 2-class head fresh.
# Dataset004 fold_0 (train 1-80 / val 81-94) -> directly comparable to the 0.870 baseline.
set +e
cd /home/youssef/projects/research/CMR-MULTI
source nnunet/env.sh
PY=/home/youssef/miniconda3/bin/python
TR=nnUNetTrainerSMP_effv2xl_RASwarm
PL=nnUNetPlansXLw
export RASWARM_CKPT=/home/youssef/projects/research/CMR-MULTI/nnunet/nnUNet_results/Dataset000_LGEgeneralist/nnUNetTrainerSMP_effv2xl__nnUNetPlans__2d/fold_0/checkpoint_final.pth
LOG=nnunet/results/logs/ras_warmstart.md
say(){ echo -e "\n$(date '+%H:%M') | $*" | tee -a "$LOG"; }

say "=========== RAS WARM-START (XL generalist -> RAS) fold_0 ==========="
say "train (batch 17, 100ep, warm-started encoder+decoder)"
nnUNetv2_train 004 2d 0 -tr $TR -p $PL >> nnunet/results/logs/ras_warm_train.log 2>&1
say "  train done (exit $?)"

say "predict held-out 81-94"
IN=nnunet/_ras_stage1_in   # already holds RAS_081..094_0000
OUT=nnunet/predictions/val/expert_RAS_xlwarm
rm -rf $OUT
nnUNetv2_predict -i $IN -o $OUT -d 004 -c 2d -tr $TR -p $PL -f 0 2>&1 | tail -2 | tee -a "$LOG"

say "score vs baseline 0.8702 (current expert):"
PYTHONPATH=. $PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import numpy as np, nibabel as nib
from pathlib import Path
from nnunet.master_rows import DATA, our_dice
def load(p): return np.asarray(nib.load(str(p)).dataobj).astype(int)
def score(d):
    P=Path(d); o={}
    for pid in range(81,95):
        pf=P/f"RAS_0{pid}.nii.gz"; gf=DATA/f"RAS_VAL/anno/LGE_RAS_0{pid}.nii.gz"
        if pf.exists(): o[pid]=our_dice(load(pf),load(gf),1)
    return o
cur=score("nnunet/predictions/val/expert_RAS"); xw=score("nnunet/predictions/val/expert_RAS_xlwarm")
print(f"{'pid':>4} {'current':>8} {'XLwarm':>8} {'delta':>7}")
for pid in range(81,95):
    c=cur.get(pid,float('nan')); w=xw.get(pid,float('nan'))
    flag=" <-- outlier" if c<0.82 else ""
    print(f"{pid:>4} {c:8.4f} {w:8.4f} {w-c:+7.4f}{flag}")
m=lambda d: float(np.mean(list(d.values())))
print(f"\nMEAN  current={m(cur):.4f}   XLwarm={m(xw):.4f}   delta={m(xw)-m(cur):+.4f}")
print("KEEP if XLwarm > 0.8702 (esp. recovers pid92 0.476).")
PYEOF
say "=========== RAS WARM-START DONE ==========="
