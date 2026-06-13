"""VALIDATE every number in LEADERBOARD.md Table A from saved predictions.

Recomputes, per model, all 3 regimes x 6 columns directly from the .nii.gz preds + GT,
then diffs against the values written in the table. Prints PASS/FAIL per cell.

  Regime ①  Ours  (OOF-40): our per-volume skip-empty Dice (compute_dsc convention)
  Regime ②  Paper (OOF-40): their dice_score (per-slice, bs=4, empty->1.0)
  Regime ③  Comp  (7-val):  their dice_score on the official 7 val patients (+pp), RAS=shared expert
  mass: SAX scar (label 3) -> grams; a = grid-refit on OOF (leakage-safe); vPCC scale-invariant; RAE on cal
  Task2 = 0.7*DSC + 0.3*(0.5*max(0,vPCC) + 0.5/(1+RAE))

Usage: PYTHONPATH=. python nnunet/validate_table.py
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, SimpleITK as sitk, nibabel as nib, pandas as pd
sys.path.insert(0, "/tmp/theirmetric")
from metrics import dice_score, mean_dice   # THEIR exact code

ROOT  = Path("/home/youssef/projects/research/CMR-MULTI")
RESB  = ROOT / "nnunet/nnUNet_results/Dataset000_LGEgeneralist"
DATA  = ROOT / "LGE_MULTI"
PREDV = ROOT / "nnunet/predictions/val"
RAS_DIR = PREDV / "expert_RAS"
DENS, SCAR = 1.05, 3
OUR_FG   = {"SAX": [1, 2, 3, 4], "2CH": [1, 2, 3], "4CH": [1, 2, 3, 4], "RAS": [1]}
THEIR_NC = {"SAX": 5, "2CH": 4, "4CH": 5, "RAS": 2}
LV = ("SAX", "2CH", "4CH")
SAX_IDS = list(range(41, 48)); RAS_IDS = list(range(81, 95))

MODELS = {
    "EffNetV2-XL":     ("nnUNetTrainerSMP_effv2xl__nnUNetPlans__2d", "cand_effv2xl_ens_pp"),
    "EffNetV2-L":      ("nnUNetTrainerSMP_effv2l__nnUNetPlans__2d",  "cand_effv2l_ens_pp"),
    "EfficientNet-B7": ("nnUNetTrainerSMP_effb7__nnUNetPlans__2d",   "cand_smpB7_ens_pp"),
    "ScarTversky":     ("nnUNetTrainerScarTversky__nnUNetPlans__2d", "cand_ensTv_pp"),
    "PlainConv 100ep": ("nnUNetTrainer_100epochs__nnUNetPlans__2d",  "cand_plainconv_pp"),
    "Hiera-tiny":      ("nnUNetTrainerHieraMedSAM2__nnUNetPlans__2d","cand_hiera_medsam2_ens_pp"),
    "SAM2.1-large":    ("nnUNetTrainerSAM2large__nnUNetPlans__2d",   "cand_sam2large_ens_pp"),
    "densenet121-RIN": ("nnUNetTrainerRIN_densenet121__nnUNetPlans__2d", "cand_dnrin_ens_pp"),
}

# EXPECTED values as written in LEADERBOARD.md (DSC, scar, vPCC, RAE, a, Task2) per regime
EXPECTED = {
 "EffNetV2-XL":    {"1": (.6981,.4112,.576,.685,.46,.6641), "2": (.7102,.4946,.576,.685,.46,.6726), "3": (.7640,.5479,.611,.878,.46,.7064)},
 "EffNetV2-L":     {"1": (.6920,.3961,.571,.692,.33,.6587), "2": (.7027,.4719,.571,.692,.33,.6662), "3": (.7726,.5723,.562,.833,.33,.7069)},
 "EfficientNet-B7":{"1": (.6923,.3995,.538,.668,.32,.6552), "2": (.7055,.4768,.538,.668,.32,.6645), "3": (.7647,.5604,.585,.824,.32,.7053)},
 "ScarTversky":    {"1": (.6736,.3624,.513,.728,.24,.6353), "2": (.6910,.4575,.513,.728,.24,.6475), "3": (.7540,.5241,.789,.710,.24,.7338)},
 "PlainConv 100ep":{"1": (.6696,.3521,.498,.718,.38,.6307), "2": (.6900,.4464,.498,.718,.38,.6450), "3": None},
 "Hiera-tiny":     {"1": (.6473,.3291,.417,.737,.30,.6020), "2": (.6660,.4182,.417,.737,.30,.6151), "3": (.7414,.4852,.545,.843,.30,.6820)},
 "SAM2.1-large":   {"1": (.6276,.2943,.374,.757,.26,.5808), "2": (.6448,.3692,.374,.757,.26,.5928), "3": (.7359,.4535,.652,.662,.26,.7032)},
 "densenet121-RIN":{"1": None, "2": None, "3": None},
}

# --------- helpers ---------
def _sitk(p): return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.int64)

def our_dice(pr, gt, classes):
    out = {}
    for c in classes:
        p, g = (pr == c), (gt == c)
        den = int(p.sum() + g.sum())
        if den == 0: continue
        out[c] = 2.0 * int((p & g).sum()) / den
    return out

def task2(dsc, vpcc, rae):
    return 0.7 * dsc + 0.3 * (0.5 * max(0.0, vpcc) + 0.5 / (1.0 + rae))

def mass_grams(path):
    img = sitk.ReadImage(str(path))
    vv = float(np.prod(img.GetSpacing())) / 1000.0
    return float((sitk.GetArrayFromImage(img).astype(np.int64) == SCAR).sum() * vv * DENS)

def fit_a(raw, true):
    nz = true > 1e-6
    grid = np.arange(0.20, 3.001, 0.01)
    def rae(p): return float(np.mean(np.abs(p[nz] - true[nz]) / true[nz]))
    return float(grid[int(np.argmin([rae(np.clip(g * raw, 0, None)) for g in grid]))])

# --------- regime computations ---------
def oof_preds(run):
    pr = defaultdict(dict)
    for k in range(5):
        for p in (RESB / run / f"fold_{k}" / "validation").glob("*.nii.gz"):
            view, pid = p.name.replace(".nii.gz", "").split("_")[0], int(p.name.replace(".nii.gz", "").split("_")[1])
            pr[pid][view] = p
    return pr

def regime_ours(run):
    pr = oof_preds(run); allv = []; scar = []
    for pid, vp in pr.items():
        for view, pp in vp.items():
            gp = DATA / f"{view}_TR" / "anno" / f"LGE_{view}_{pid:03d}.nii.gz"
            if not gp.exists(): continue
            g = np.asarray(nib.load(str(gp)).dataobj).astype(int)
            p = np.asarray(nib.load(str(pp)).dataobj).astype(int)
            if p.shape != g.shape: continue
            for c, v in our_dice(p, g, OUR_FG[view]).items():
                allv.append(v)
                if c == SCAR and view in LV: scar.append(v)
    return float(np.mean(allv)), float(np.mean(scar))

def their_view(pred_dir, view, nc, ids, gt_tr=True):
    sp, sg = [], []
    for pid in ids:
        pf = pred_dir / f"{view}_{pid:03d}.nii.gz"
        gf = (DATA / f"{view}_TR" / "anno" / f"LGE_{view}_{pid:03d}.nii.gz") if gt_tr \
             else (DATA / f"{view}_VAL" / "anno" / f"LGE_{view}_{pid:03d}.nii.gz")
        if not pf.exists() or not gf.exists(): continue
        p, g = _sitk(pf), _sitk(gf)
        if p.shape != g.shape: continue
        for d in range(g.shape[0]): sp.append(p[d]); sg.append(g[d])
    if not sp: return None
    P, G = torch.from_numpy(np.stack(sp)), torch.from_numpy(np.stack(sg))
    acc = [0.0] * nc; nb = 0
    for i in range(0, P.shape[0], 4):
        dc = dice_score(P[i:i+4], G[i:i+4], nc)
        for c in range(nc): acc[c] += dc[c]
        nb += 1
    return [v / nb for v in acc]

def regime_paper(run):
    # their metric over OOF: gather each view's OOF preds into a temp per-view dir-like map
    pr = oof_preds(run)
    fg, scar = [], []
    for view, nc in THEIR_NC.items():
        sp, sg = [], []
        for pid in sorted(pr):          # sorted-pid order = same batching as score_their_metric (bs=4 is order-dependent)
            vp = pr[pid]
            if view not in vp: continue
            gf = DATA / f"{view}_TR" / "anno" / f"LGE_{view}_{pid:03d}.nii.gz"
            if not gf.exists(): continue
            p, g = _sitk(vp[view]), _sitk(gf)
            if p.shape != g.shape: continue
            for d in range(g.shape[0]): sp.append(p[d]); sg.append(g[d])
        if not sp: continue
        P, G = torch.from_numpy(np.stack(sp)), torch.from_numpy(np.stack(sg))
        acc = [0.0]*nc; nb=0
        for i in range(0, P.shape[0], 4):
            dc = dice_score(P[i:i+4], G[i:i+4], nc)
            for c in range(nc): acc[c]+=dc[c]
            nb+=1
        per=[v/nb for v in acc]; fg.append(mean_dice(per))
        if nc > SCAR and view in LV: scar.append(per[SCAR])
    return float(np.mean(fg)), float(np.mean(scar))

def mass_oof(run):
    sq = pd.read_excel(DATA / "dataset_lge.xlsx", sheet_name="SAX")
    tmap = dict(zip(sq["编号"].astype(int), sq["Scar_Quality"].astype(float)))
    pr = oof_preds(run); pids = sorted(p for p in pr if "SAX" in pr[p])
    raw = np.array([mass_grams(pr[p]["SAX"]) for p in pids]); true = np.array([tmap[p] for p in pids])
    a = fit_a(raw, true); nz = true > 1e-6
    vpcc = float(np.corrcoef(raw, true)[0, 1])
    rae_cal = float(np.mean(np.abs(np.clip(a*raw,0,None)[nz] - true[nz]) / true[nz]))
    return vpcc, rae_cal, a

def regime_comp(cand, a):
    cdir = PREDV / cand
    if not cdir.exists(): return None
    fg, scar = [], []
    for view, nc in THEIR_NC.items():
        per = their_view(RAS_DIR if view == "RAS" else cdir, view, nc,
                         RAS_IDS if view == "RAS" else SAX_IDS, gt_tr=False)
        if per is None: continue
        fg.append(mean_dice(per))
        if nc > SCAR and view in LV: scar.append(per[SCAR])
    dsc = float(np.mean(fg)); scarv = float(np.mean(scar))
    sq = pd.read_excel(DATA / "dataset_lge_valid.xlsx", sheet_name="SAX")
    tmap = dict(zip(sq["编号"].astype(int), sq["Scar_Quality"].astype(float)))
    pred, true = [], []
    for pid in SAX_IDS:
        pf = cdir / f"SAX_{pid:03d}.nii.gz"
        if not pf.exists(): continue
        pred.append(a * mass_grams(pf)); true.append(tmap[pid])
    pred, true = np.array(pred), np.array(true); nz = true > 1e-6
    vpcc = float(np.corrcoef(pred, true)[0, 1]); rae = float(np.mean(np.abs(pred[nz]-true[nz])/true[nz]))
    return dsc, scarv, vpcc, rae

# --------- run + diff ---------
def chk(name, regime, got, exp):
    cols = ["DSC", "scar", "vPCC", "RAE", "a", "Task2"]
    tol  = [.001, .001, .0015, .0015, .011, .001]
    if exp is None:
        print(f"  {name:<16} {regime}: " + "  ".join(f"{c}={got[i]:.4f}" for i, c in enumerate(cols)) + "   [NEW — no table value yet]")
        return True
    ok = True
    line = f"  {name:<16} {regime}: "
    for i, c in enumerate(cols):
        d = abs(got[i] - exp[i]); good = d <= tol[i]
        ok &= good
        line += f"{c}={got[i]:.4f}{'' if good else f'≠{exp[i]:.4f}!!'}  "
    print(line + ("PASS" if ok else "  <<< MISMATCH"))
    return ok

def main():
    allok = True
    for name, (run, cand) in MODELS.items():
        d1, s1 = regime_ours(run)
        d2, s2 = regime_paper(run)
        vp, rae, a = mass_oof(run)
        t1 = task2(d1, vp, rae); t2 = task2(d2, vp, rae)
        allok &= chk(name, "①", (d1, s1, vp, rae, a, t1), EXPECTED[name]["1"])
        allok &= chk(name, "②", (d2, s2, vp, rae, a, t2), EXPECTED[name]["2"])
        comp = regime_comp(cand, a)
        if comp is None:
            print(f"  {name:<16} ③: (val preds not present: {cand})")
        else:
            d3, s3, vp3, rae3 = comp; t3 = task2(d3, vp3, rae3)
            allok &= chk(name, "③", (d3, s3, vp3, rae3, a, t3), EXPECTED[name]["3"])
        print()
    print("================", "ALL CELLS VALIDATED ✓" if allok else "MISMATCHES FOUND — SEE ABOVE", "================")

if __name__ == "__main__":
    main()
