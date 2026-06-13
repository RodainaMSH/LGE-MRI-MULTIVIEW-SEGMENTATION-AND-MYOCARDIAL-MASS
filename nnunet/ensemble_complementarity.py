"""
Ensemble complementarity analysis (CPU, GPU-free) for the 3 generalists.
Uses existing argmax OOF preds (no probabilities saved -> prob-averaging needs GPU).

Answers, per class:
  1. Per-model OOF DSC (sanity, = leaderboard regime ①).
  2. ORACLE ceiling = best-of-3 picked per (case,class). Gap vs best single = max ensemble upside.
  3. WIN distribution = how often each model is *uniquely* best (spread -> ensemble helps; one-dominates -> it won't).
  4. MAJORITY-VOTE ensemble = per-voxel mode of the 3 label maps (tie -> XL). Scored under our skip-empty metric.
     (For scar this is conservative -> may cut over-seg, our mass weakness.)
"""
import sys, numpy as np, nibabel as nib
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified_eval.compute_dsc import dice_np, VIEW_FG, CLASS_NAME, DATA

RES = Path("nnunet/nnUNet_results/Dataset000_LGEgeneralist")
MODELS = {"XL": "nnUNetTrainerSMP_effv2xl", "L": "nnUNetTrainerSMP_effv2l", "B7": "nnUNetTrainerSMP_effb7"}
def anno(view, pid): return DATA / f"{view}_TR" / "anno" / f"LGE_{view}_{pid:03d}.nii.gz"
def load(p): return np.asarray(nib.load(str(p)).dataobj).astype(int)

# gather OOF preds: case -> {model: path}
preds = defaultdict(dict)
for mk, tr in MODELS.items():
    d = RES / f"{tr}__nnUNetPlans__2d"
    for f in range(5):
        for p in (d / f"fold_{f}" / "validation").glob("*.nii.gz"):
            preds[p.stem][mk] = p
cases = sorted(k for k, v in preds.items() if len(v) == 3)
print(f"cases with all 3 models: {len(cases)}\n")

# per (case,class) dice for each model + majority-vote
# dice[class][model] = list; oracle[class]=list of best; wins[class][model]=count unique best
dice = defaultdict(lambda: defaultdict(list))
oracle = defaultdict(list); mv_dice = defaultdict(list); single_for_mv = defaultdict(list)
wins = defaultdict(lambda: defaultdict(float)); spreads = defaultdict(list)

for case in cases:
    view, pid = case.split("_")[0], int(case.split("_")[1].split(".")[0])
    gt = load(anno(view, pid))
    arr = {mk: load(preds[case][mk]) for mk in MODELS}
    # majority vote per voxel: where >=2 agree take that; else XL
    stack = np.stack([arr["XL"], arr["L"], arr["B7"]])
    mv = arr["XL"].copy()
    # for each class label present, count votes
    flat = stack.reshape(3, -1)
    out = np.full(flat.shape[1], -1, dtype=int)
    for lab in range(5):
        votes = (flat == lab).sum(0)
        out[votes >= 2] = lab
    out[out == -1] = flat[0][out == -1]  # tie (all differ) -> XL
    mv = out.reshape(arr["XL"].shape)

    dmv = dice_np(mv, gt, VIEW_FG[view])
    dper = {mk: dice_np(arr[mk], gt, VIEW_FG[view]) for mk in MODELS}
    for c in VIEW_FG[view]:
        vals = {mk: dper[mk].get(c) for mk in MODELS if c in dper[mk]}
        if not vals: continue
        for mk, v in vals.items(): dice[c][mk].append(v)
        best = max(vals.values()); oracle[c].append(best)
        # unique best?
        topmks = [mk for mk, v in vals.items() if abs(v - best) < 1e-9]
        for mk in topmks: wins[c][mk] += 1.0 / len(topmks)
        spreads[c].append(best - min(vals.values()))
        if c in dmv:
            mv_dice[c].append(dmv[c]); single_for_mv[c].append(vals.get("XL", np.nan))

def mean(x): return float(np.mean(x)) if len(x) else float("nan")

print(f"{'class':<10} {'XL':>7} {'L':>7} {'B7':>7} | {'ORACLE':>7} {'gap':>6} | {'MAJVOTE':>7} {'vsXL':>7} | {'wins XL/L/B7':>16} {'spread':>7}")
order = [1, 2, 3, 4]
allmodel = {mk: [] for mk in MODELS}; allora = []; allmv = []; allmvxl = []
for c in order:
    if not dice[c]: continue
    xs = {mk: mean(dice[c][mk]) for mk in MODELS}
    ora = mean(oracle[c]); mv = mean(mv_dice[c]); mvxl = mean(single_for_mv[c])
    n = len(oracle[c]); w = wins[c]
    wstr = f"{w['XL']:.0f}/{w['L']:.0f}/{w['B7']:.0f}"
    best_single = max(xs.values())
    print(f"{CLASS_NAME[c]:<10} {xs['XL']:7.4f} {xs['L']:7.4f} {xs['B7']:7.4f} | "
          f"{ora:7.4f} {ora-best_single:+6.3f} | {mv:7.4f} {mv-mvxl:+7.4f} | {wstr:>16} {mean(spreads[c]):7.4f}")
    for mk in MODELS: allmodel[mk] += dice[c][mk]
    allora += oracle[c]; allmv += mv_dice[c]; allmvxl += single_for_mv[c]

print(f"\n{'ALL-fg':<10} {mean(allmodel['XL']):7.4f} {mean(allmodel['L']):7.4f} {mean(allmodel['B7']):7.4f} | "
      f"{mean(allora):7.4f} {mean(allora)-max(mean(allmodel[m]) for m in MODELS):+6.3f} | {mean(allmv):7.4f} {mean(allmv)-mean(allmvxl):+7.4f}")
print("\nREAD: ORACLE-gap = upside if we could pick the best model per case (ensemble ceiling).")
print("      MAJVOTE vsXL = what the FREE per-voxel majority-vote ensemble actually gains/loses now.")
print("      wins spread across XL/L/B7 -> models are complementary (ensemble helps); concentrated -> it won't.")
