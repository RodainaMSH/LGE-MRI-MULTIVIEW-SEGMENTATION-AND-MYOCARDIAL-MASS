#!/usr/bin/env bash
# Multi-arch SOFT (probability-averaged) OOF ensemble. Leakage-clean: for each fold k, each arch's
# fold-k model predicts ONLY fold-k's val set (the OOF), then nnUNetv2_ensemble averages the 4 archs'
# softmax probs and argmaxes. Output mirrors a normal run dir so oof_eval_run.py scores it directly.
set +e
cd /home/youssef/projects/research/CMR-MULTI
source nnunet/env.sh
ARCHS="nnUNetTrainerSMP_effv2xl nnUNetTrainerSMP_densenet264 nnUNetTrainerSMP_effv2l nnUNetTrainerSMP_effb7"
SP=/tmp/ens/probs
RB=nnunet/nnUNet_results/Dataset000_LGEgeneralist
ENS=$RB/ENS_soft4__nnUNetPlans__2d
rm -rf "$ENS" "$SP"

# stage each fold's val images (leakage anchor: a case is predicted only by models that held it out)
python - <<'PY'
import json,os
s=json.load(open('nnunet/nnUNet_preprocessed/Dataset000_LGEgeneralist/splits_final.json'))
for k in range(5):
    d=f'/tmp/ens/imgs_f{k}'; os.makedirs(d,exist_ok=True)
    for c in s[k]['val']:
        src=os.path.abspath(f'nnunet/nnUNet_raw/Dataset000_LGEgeneralist/imagesTr/{c}_0000.nii.gz')
        dst=f'{d}/{c}_0000.nii.gz'
        if os.path.exists(src) and not os.path.islink(dst): os.symlink(src,dst)
    print(f'fold {k}: staged {len(s[k]["val"])} val images')
PY

for k in 0 1 2 3 4; do
  for A in $ARCHS; do
    nnUNetv2_predict -i /tmp/ens/imgs_f$k -o $SP/$A/fold_$k -d 0 -c 2d -tr $A -f $k \
        --save_probabilities -npp 2 -nps 2 > /dev/null 2>&1
    echo "  predicted $A fold $k (exit $?)"
  done
  mkdir -p $ENS/fold_$k/validation
  nnUNetv2_ensemble \
    -i $SP/nnUNetTrainerSMP_effv2xl/fold_$k $SP/nnUNetTrainerSMP_densenet264/fold_$k \
       $SP/nnUNetTrainerSMP_effv2l/fold_$k $SP/nnUNetTrainerSMP_effb7/fold_$k \
    -o $ENS/fold_$k/validation -np 2 > /dev/null 2>&1
    n=$(ls $ENS/fold_$k/validation/*.nii.gz 2>/dev/null | wc -l)
  echo "fold $k ENSEMBLED -> $n cases"
done
echo "ENSEMBLE_DONE"
