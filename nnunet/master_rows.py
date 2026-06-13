"""Produce the new master Table-A rows: per-VIEW DSC (SAX/2CH/4CH/RAS) + 4-view DSC + scar + mass + Task2,
for every model x 3 regimes. DSC is now the mean of the 4 per-view DSCs (RAS = shared expert).
scar is UNCHANGED (RAS has no scar class). Task2 recomputed with the 4-view DSC.

Usage: PYTHONPATH=. python nnunet/master_rows.py
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, SimpleITK as sitk, nibabel as nib, pandas as pd
sys.path.insert(0, "/tmp/theirmetric")
from metrics import dice_score, mean_dice

ROOT = Path("/home/youssef/projects/research/CMR-MULTI")
RESB = ROOT / "nnunet/nnUNet_results/Dataset000_LGEgeneralist"
DATA = ROOT / "LGE_MULTI"
PREDV = ROOT / "nnunet/predictions/val"
RAS_EXP = PREDV / "expert_RAS_xlwarm"   # ADOPTED 2026-06-10: XL cardiac warm-start fold_0 (0.892 > 0.870 from-scratch). Old expert_RAS kept for revert.
DENS, SCAR = 1.05, 3
OUR_FG = {"SAX": [1, 2, 3, 4], "2CH": [1, 2, 3], "4CH": [1, 2, 3, 4]}
THEIR_NC = {"SAX": 5, "2CH": 4, "4CH": 5}
LV = ["SAX", "2CH", "4CH"]
SAX_IDS = list(range(41, 48)); RAS_IDS = list(range(81, 95))

# name -> (oof run, val cand dir or None)
MODELS = [
    ("EffNetV2-XL",     "nnUNetTrainerSMP_effv2xl__nnUNetPlans__2d",   "cand_effv2xl_ens_pp"),
    ("EffNetV2-L",      "nnUNetTrainerSMP_effv2l__nnUNetPlans__2d",    "cand_effv2l_ens_pp"),
    ("EfficientNet-B7", "nnUNetTrainerSMP_effb7__nnUNetPlans__2d",     "cand_smpB7_ens_pp"),
    ("DenseNet-264",    "nnUNetTrainerSMP_densenet264__nnUNetPlans__2d", "cand_dn264_ens_pp"),
    ("densenet121-RIN", "nnUNetTrainerRIN_densenet121__nnUNetPlans__2d", "cand_dnrin_ens_pp"),
    ("ScarTversky",     "nnUNetTrainerScarTversky__nnUNetPlans__2d",   "cand_ensTv_pp"),
    ("PlainConv 100ep", "nnUNetTrainer_100epochs__nnUNetPlans__2d",    "cand_plainconv_pp"),
    ("Hiera-tiny",      "nnUNetTrainerHieraMedSAM2__nnUNetPlans__2d",  "cand_hiera_medsam2_ens_pp"),
    ("SAM2.1-large",    "nnUNetTrainerSAM2large__nnUNetPlans__2d",     "cand_sam2large_ens_pp"),
]

def _sitk(p): return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.int64)
def our_dice(pr, gt, c):
    p, g = (pr == c), (gt == c); den = int(p.sum() + g.sum())
    return None if den == 0 else 2.0 * int((p & g).sum()) / den
def mass_grams(p):
    img = sitk.ReadImage(str(p)); vv = float(np.prod(img.GetSpacing())) / 1000.0
    return float((sitk.GetArrayFromImage(img).astype(np.int64) == SCAR).sum() * vv * DENS)
def massscore(vp, rae): return 0.5 * max(0.0, vp) + 0.5 / (1.0 + rae)
def task2(dsc, vp, rae): return 0.7 * dsc + 0.3 * massscore(vp, rae)

def oof_preds(run):
    pr = defaultdict(dict)
    for k in range(5):
        for p in (RESB / run / f"fold_{k}" / "validation").glob("*.nii.gz"):
            v, pid = p.name.replace(".nii.gz", "").split("_")[0], int(p.name.replace(".nii.gz", "").split("_")[1])
            pr[pid][v] = p
    return pr

# shared RAS (held-out 81-94): our + their
def ras_shared():
    ours, sp, sg = [], [], []
    for pid in RAS_IDS:
        pf = RAS_EXP / f"RAS_{pid:03d}.nii.gz"; gf = DATA / "RAS_VAL/anno" / f"LGE_RAS_{pid:03d}.nii.gz"
        if not pf.exists(): continue
        d = our_dice(np.asarray(nib.load(str(pf)).dataobj).astype(int), np.asarray(nib.load(str(gf)).dataobj).astype(int), 1)
        if d is not None: ours.append(d)
        a, b = _sitk(pf), _sitk(gf)
        for z in range(b.shape[0]): sp.append(a[z]); sg.append(b[z])
    P, G = torch.from_numpy(np.stack(sp)), torch.from_numpy(np.stack(sg)); acc = [0., 0.]; nb = 0
    for i in range(0, P.shape[0], 4):
        dc = dice_score(P[i:i+4], G[i:i+4], 2)
        for c in range(2): acc[c] += dc[c]
        nb += 1
    return float(np.mean(ours)), acc[1] / nb
RAS_OUR, RAS_THEIR = ras_shared()

def our_view_scar(run):  # regime ① : per-view fg + scar over SAX/2CH/4CH
    pr = oof_preds(run); vv = defaultdict(list); scar = []
    for pid, vp in pr.items():
        for v, pp in vp.items():
            if v not in OUR_FG: continue
            gp = DATA / f"{v}_TR/anno" / f"LGE_{v}_{pid:03d}.nii.gz"
            if not gp.exists(): continue
            g = np.asarray(nib.load(str(gp)).dataobj).astype(int); p = np.asarray(nib.load(str(pp)).dataobj).astype(int)
            if p.shape != g.shape: continue
            for c in OUR_FG[v]:
                d = our_dice(p, g, c)
                if d is None: continue
                vv[v].append(d)
                if c == SCAR: scar.append(d)
    view = {v: float(np.mean(vv[v])) for v in LV}; view["RAS"] = RAS_OUR
    return view, float(np.mean(scar))

def their_view_scar(run=None, cand=None):  # regime ② (oof) or ③ (val)
    view, scar = {}, []
    for v in LV:
        nc = THEIR_NC[v]; sp, sg = [], []
        if cand:  # ③ val
            for pid in SAX_IDS:
                pf = PREDV / cand / f"{v}_{pid:03d}.nii.gz"; gf = DATA / f"{v}_VAL/anno" / f"LGE_{v}_{pid:03d}.nii.gz"
                if not pf.exists() or not gf.exists(): continue
                a, b = _sitk(pf), _sitk(gf)
                if a.shape != b.shape: continue
                for z in range(b.shape[0]): sp.append(a[z]); sg.append(b[z])
        else:     # ② oof
            pr = oof_preds(run)
            for pid in sorted(pr):
                if v not in pr[pid]: continue
                gf = DATA / f"{v}_TR/anno" / f"LGE_{v}_{pid:03d}.nii.gz"
                if not gf.exists(): continue
                a, b = _sitk(pr[pid][v]), _sitk(gf)
                if a.shape != b.shape: continue
                for z in range(b.shape[0]): sp.append(a[z]); sg.append(b[z])
        if not sp: view[v] = float("nan"); continue
        P, G = torch.from_numpy(np.stack(sp)), torch.from_numpy(np.stack(sg)); acc = [0.]*nc; nb = 0
        for i in range(0, P.shape[0], 4):
            dc = dice_score(P[i:i+4], G[i:i+4], nc)
            for c in range(nc): acc[c] += dc[c]
            nb += 1
        per = [x/nb for x in acc]; view[v] = mean_dice(per)
        if nc > SCAR: scar.append(per[SCAR])
    view["RAS"] = RAS_THEIR
    return view, float(np.mean(scar))

def mass_oof(run):
    sq = pd.read_excel(DATA / "dataset_lge.xlsx", sheet_name="SAX")
    tmap = dict(zip(sq["编号"].astype(int), sq["Scar_Quality"].astype(float)))
    pr = oof_preds(run); pids = sorted(p for p in pr if "SAX" in pr[p])
    raw = np.array([mass_grams(pr[p]["SAX"]) for p in pids]); true = np.array([tmap[p] for p in pids]); nz = true > 1e-6
    grid = np.arange(0.20, 3.001, 0.01)
    rae = lambda pp: float(np.mean(np.abs(pp[nz]-true[nz])/true[nz]))
    a = float(grid[int(np.argmin([rae(np.clip(g*raw,0,None)) for g in grid]))])
    return float(np.corrcoef(raw, true)[0,1]), rae(np.clip(a*raw,0,None)), a

def mass_val(cand, a):
    sq = pd.read_excel(DATA / "dataset_lge_valid.xlsx", sheet_name="SAX")
    tmap = dict(zip(sq["编号"].astype(int), sq["Scar_Quality"].astype(float)))
    pred, true = [], []
    for pid in SAX_IDS:
        pf = PREDV / cand / f"SAX_{pid:03d}.nii.gz"
        if not pf.exists(): continue
        pred.append(a*mass_grams(pf)); true.append(tmap[pid])
    pred, true = np.array(pred), np.array(true); nz = true > 1e-6
    return float(np.corrcoef(pred, true)[0,1]), float(np.mean(np.abs(pred[nz]-true[nz])/true[nz]))

def fmt(x): return "—" if (x != x) else f"{x:.4f}"
def m(view): return float(np.nanmean([view[v] for v in ["SAX","2CH","4CH","RAS"]]))

if __name__ == "__main__":
    print(f"RAS shared: our {RAS_OUR:.4f} / their {RAS_THEIR:.4f}\n")
    print(f"{'model':<16}{'reg':<4}{'SAX':>8}{'2CH':>8}{'4CH':>8}{'RAS':>8}{'DSC':>8}{'scar':>8}{'vPCC':>7}{'RAE':>7}{'a':>6}{'Mass':>8}{'Task2':>8}")
    for name, run, cand in MODELS:
        v1, s1 = our_view_scar(run); v2, s2 = their_view_scar(run=run)
        vp, rae, a = mass_oof(run)
        d1, d2 = m(v1), m(v2); ms = massscore(vp, rae)
        print(f"{name:<16}{'①':<4}{v1['SAX']:>8.4f}{v1['2CH']:>8.4f}{v1['4CH']:>8.4f}{v1['RAS']:>8.4f}{d1:>8.4f}{s1:>8.4f}{vp:>7.3f}{rae:>7.3f}{a:>6.2f}{ms:>8.4f}{task2(d1,vp,rae):>8.4f}")
        print(f"{'':<16}{'②':<4}{v2['SAX']:>8.4f}{v2['2CH']:>8.4f}{v2['4CH']:>8.4f}{v2['RAS']:>8.4f}{d2:>8.4f}{s2:>8.4f}{vp:>7.3f}{rae:>7.3f}{a:>6.2f}{ms:>8.4f}{task2(d2,vp,rae):>8.4f}")
        if cand:
            v3, s3 = their_view_scar(cand=cand); d3 = m(v3); vpv, raev = mass_val(cand, a); ms3 = massscore(vpv, raev)
            print(f"{'':<16}{'③':<4}{v3['SAX']:>8.4f}{v3['2CH']:>8.4f}{v3['4CH']:>8.4f}{v3['RAS']:>8.4f}{d3:>8.4f}{s3:>8.4f}{vpv:>7.3f}{raev:>7.3f}{a:>6.2f}{ms3:>8.4f}{task2(d3,vpv,raev):>8.4f}")
        else:
            print(f"{'':<16}{'③':<4}{'—':>8}{'—':>8}{'—':>8}{'—':>8}{'—':>8}{'—':>8}{'—':>7}{'—':>7}{'—':>6}{'—':>8}{'—':>8}")
