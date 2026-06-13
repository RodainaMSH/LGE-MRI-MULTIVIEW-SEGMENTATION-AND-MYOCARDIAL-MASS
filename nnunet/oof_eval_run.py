"""OOF evaluation for ANY nnU-Net trainer run on Dataset000 (5-fold).
Reports OOF DSC (overall + per-class, esp. scar) and refits the scale-only SAX scar-mass
calibration. Compares to the vanilla baseline (OOF DSC 0.670 / scar 0.352, scale 0.380).

Usage: PYTHONPATH=. python nnunet/oof_eval_run.py <run_dirname>
  e.g. PYTHONPATH=. python nnunet/oof_eval_run.py nnUNetTrainerScarTversky__nnUNetPlans__2d
"""
from __future__ import annotations
import sys, json
from collections import defaultdict
from pathlib import Path
import nibabel as nib, numpy as np, pandas as pd

from common.metrics import vpcc, rae
from unified_eval.compute_dsc import dice_np, VIEW_FG, CLASS_NAME

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "LGE_MULTI"
RESBASE = ROOT / "nnunet" / "nnUNet_results" / "Dataset000_LGEgeneralist"
DENS, SCAR = 1.05, 3
BASELINE = {"oof_dsc": 0.670, "oof_scar": 0.352, "scale": 0.380}


def scar_quality(split):
    f = "dataset_lge.xlsx" if split == "TR" else "dataset_lge_valid.xlsx"
    df = pd.read_excel(DATA / f, sheet_name="SAX")
    return dict(zip(df["编号"].astype(int), df["Scar_Quality"].astype(float)))


def sax_scar_mass(path):
    nii = nib.load(str(path)); a = np.asarray(nii.dataobj).astype(int)
    vv = float(np.prod(nii.header.get_zooms()[:3])) / 1000.0
    return float((a == SCAR).sum() * vv * DENS)


def anno(view, pid):
    return DATA / f"{view}_TR" / "anno" / f"LGE_{view}_{pid:03d}.nii.gz"


def main():
    run = sys.argv[1]
    RES = RESBASE / run
    preds = defaultdict(dict)
    done = 0
    for k in range(5):
        vdir = RES / f"fold_{k}" / "validation"
        if not vdir.exists():
            continue
        done += 1
        for p in vdir.glob("*.nii.gz"):
            stem = p.name.replace(".nii.gz", "")
            view, pid = stem.split("_")[0], int(stem.split("_")[1])
            preds[pid][view] = p
    print(f"RUN: {run}\nfolds with validation: {done}/5   OOF patients: {len(preds)}")

    per_class = defaultdict(list); allv = []
    scar_per = defaultdict(list)  # SAX-only scar per the dice_np metric
    for pid, vp in preds.items():
        for view, pp in vp.items():
            gp = anno(view, pid)
            if not gp.exists():
                continue
            gt = np.asarray(nib.load(str(gp)).dataobj).astype(int)
            pr = np.asarray(nib.load(str(pp)).dataobj).astype(int)
            if pr.shape != gt.shape:
                continue
            for c, v in dice_np(pr, gt, VIEW_FG[view]).items():
                per_class[c].append(v); allv.append(v)
    oof_dsc = float(np.mean(allv)) if allv else float("nan")
    print(f"\nOOF CV DSC overall = {oof_dsc:.4f}   (baseline vanilla {BASELINE['oof_dsc']:.4f}, delta {oof_dsc-BASELINE['oof_dsc']:+.4f})")
    for c in sorted(per_class):
        m = float(np.mean(per_class[c]))
        extra = f"   (baseline {BASELINE['oof_scar']:.4f}, delta {m-BASELINE['oof_scar']:+.4f})" if CLASS_NAME[c] == "scar" else ""
        print(f"   {CLASS_NAME[c]:>9}: {m:.4f}{extra}")

    # scale-only SAX scar-mass calibration refit
    sq = scar_quality("TR")
    pids = [p for p in sorted(preds) if "SAX" in preds[p]]
    if len(pids) >= 5:
        pred = np.array([sax_scar_mass(preds[p]["SAX"]) for p in pids])
        true = np.array([sq[p] for p in pids])
        grid = np.arange(0.20, 3.001, 0.01)
        a = float(grid[int(np.argmin([rae(np.clip(g*pred, 0, None), true) for g in grid]))])
        cal = np.clip(a*pred, 0, None)
        print(f"\nREFIT scale-only calibration on {len(pids)} OOF SAX: a={a:.3f}  (baseline a={BASELINE['scale']:.3f})")
        print(f"  OOF vPCC {vpcc(pred,true):.3f}   RAE {rae(pred,true):.3f}->{rae(cal,true):.3f}")
    print(f"\nKEEP? need OOF DSC delta and/or scar delta to clear fold-noise (>+0.02). Decide on this, not fold-0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
