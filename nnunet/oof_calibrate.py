"""From the 5-fold generalist (Dataset000), assemble OUT-OF-FOLD predictions and:
  1. report honest cross-val DSC (overall + per class, LV views),
  2. fit the SAX scar-mass CALIBRATION (Scar_Quality ~= a*pred + b) on the 40 train OOF,
  3. apply it to the 7 official val and report scar-mass vPCC/RAE + clinical term BEFORE vs AFTER.

The OOF preds are nnU-Net's own per-fold validation outputs (fold_k/validation/). The 7-val SAX preds
come from the preserved official-split generalist (fold_0_officialval/validation/). Official scar mass =
SAX-only voxels * voxel_vol * 1.05 == the xlsx `Scar_Quality` column (verified).

Run after the 5 folds finish (safe to run partially):
    PYTHONPATH=. python nnunet/oof_calibrate.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from common.metrics import vpcc, rae
from unified_eval.compute_dsc import dice_np, VIEW_FG, CLASS_NAME

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "LGE_MULTI"
RES = ROOT / "nnunet" / "nnUNet_results" / "Dataset000_LGEgeneralist" / "nnUNetTrainer_100epochs__nnUNetPlans__2d"
DENS, SCAR = 1.05, 3


def scar_quality(split: str) -> dict[int, float]:
    f = "dataset_lge.xlsx" if split == "TR" else "dataset_lge_valid.xlsx"
    df = pd.read_excel(DATA / f, sheet_name="SAX")
    return dict(zip(df["编号"].astype(int), df["Scar_Quality"].astype(float)))


def sax_scar_mass(path: Path) -> float:
    nii = nib.load(str(path))
    a = np.asarray(nii.dataobj).astype(int)
    vv = float(np.prod(nii.header.get_zooms()[:3])) / 1000.0
    return float((a == SCAR).sum() * vv * DENS)


def anno(view: str, split: str, pid: int) -> Path:
    return DATA / f"{view}_{split}" / "anno" / f"LGE_{view}_{pid:03d}.nii.gz"


def gather_oof() -> dict[int, dict[str, Path]]:
    preds: dict[int, dict[str, Path]] = defaultdict(dict)
    for k in range(5):
        vdir = RES / f"fold_{k}" / "validation"
        if not vdir.exists():
            continue
        for p in vdir.glob("*.nii.gz"):
            stem = p.name.replace(".nii.gz", "")
            view, pid = stem.split("_")[0], int(stem.split("_")[1])
            preds[pid][view] = p
    return preds


def clin(v: float, r: float) -> float:
    return 0.5 * max(0.0, v) + 0.5 / (1.0 + r)


def main() -> int:
    oof = gather_oof()
    done = sum((RES / f"fold_{k}" / "validation").exists() for k in range(5))
    print(f"folds with validation preds: {done}/5   OOF patients: {len(oof)}")
    out: dict = {"folds_done": done, "n_oof_patients": len(oof)}

    # 1) OOF cross-val DSC (LV views)
    per_class: dict[int, list[float]] = defaultdict(list)
    allv: list[float] = []
    for pid, vp in oof.items():
        for view, pp in vp.items():
            gp = anno(view, "TR", pid)
            if not gp.exists():
                continue
            gt = np.asarray(nib.load(str(gp)).dataobj).astype(int)
            pr = np.asarray(nib.load(str(pp)).dataobj).astype(int)
            if pr.shape != gt.shape:
                continue
            for c, v in dice_np(pr, gt, VIEW_FG[view]).items():
                per_class[c].append(v)
                allv.append(v)
    oof_dsc = float(np.mean(allv)) if allv else float("nan")
    print(f"\nOOF CV DSC (LV views, {len(oof)} train pts): overall = {oof_dsc:.4f}")
    pc = {}
    for c in sorted(per_class):
        pc[CLASS_NAME[c]] = float(np.mean(per_class[c]))
        print(f"   {CLASS_NAME[c]:>9}: {pc[CLASS_NAME[c]]:.4f}")
    out["oof_dsc_overall"] = oof_dsc
    out["oof_dsc_per_class"] = pc

    # 2) SCALE-ONLY calibration on OOF SAX scar mass.
    # NOTE: an intercept is WRONG here — it adds mass to ~zero-scar patients and RAE divides by
    # true, blowing up. vPCC is scale-invariant, so calibration only moves RAE. We pick the single
    # scale `a` that MINIMIZES OOF RAE (grid search), then apply it to the held-out 7 val.
    sq_tr = scar_quality("TR")
    pids = [p for p in sorted(oof) if "SAX" in oof[p]]
    if len(pids) >= 5:
        pred = np.array([sax_scar_mass(oof[p]["SAX"]) for p in pids])
        true = np.array([sq_tr[p] for p in pids])
        grid = np.arange(0.20, 3.001, 0.01)
        raes = [rae(np.clip(g * pred, 0, None), true) for g in grid]
        a = float(grid[int(np.argmin(raes))])
        cal = np.clip(a * pred, 0, None)
        print(f"\nSCALE-ONLY calibration on {len(pids)} OOF train: Scar_Quality ~= {a:.3f}*pred")
        print(f"  OOF vPCC {vpcc(pred, true):.3f} (scale-invariant)  "
              f"RAE {rae(pred, true):.3f}->{rae(cal, true):.3f}  (a=1 means no gain)")
        out["calibration"] = {"scale_a": a, "oof_rae_raw": rae(pred, true), "oof_rae_cal": rae(cal, true),
                              "oof_vpcc": vpcc(pred, true)}

        # 3) held-out test on the 7 official val (scale fit on train, applied to val)
        valdir = RES / "fold_0_officialval" / "validation"
        if valdir.exists():
            sq_val = scar_quality("VAL")
            vpids = sorted(sq_val)
            vpred = np.array([sax_scar_mass(valdir / f"SAX_{p:03d}.nii.gz") for p in vpids])
            vtrue = np.array([sq_val[p] for p in vpids])
            vcal = np.clip(a * vpred, 0, None)
            v0, r0 = vpcc(vpred, vtrue), rae(vpred, vtrue)
            v1, r1 = vpcc(vcal, vtrue), rae(vcal, vtrue)
            print("\nHELD-OUT 7 val (calibration fit on 40 train, applied to val):")
            print(f"  raw : vPCC {v0:.3f}  RAE {r0:.3f}  clin {clin(v0, r0):.3f}")
            print(f"  CAL : vPCC {v1:.3f}  RAE {r1:.3f}  clin {clin(v1, r1):.3f}")
            print(f"  => Task2 @ our DSC {oof_dsc:.3f}: raw {0.7*oof_dsc+0.3*clin(v0,r0):.4f} | "
                  f"cal {0.7*oof_dsc+0.3*clin(v1,r1):.4f}")
            out["val_calibration_test"] = {
                "raw": {"vpcc": v0, "rae": r0, "clin": clin(v0, r0)},
                "cal": {"vpcc": v1, "rae": r1, "clin": clin(v1, r1)},
            }
    else:
        print("\n(need >=5 OOF SAX patients for calibration; run after more folds finish)")

    (ROOT / "nnunet" / "results" / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "nnunet" / "results" / "logs" / "oof_calibrate.json").write_text(json.dumps(out, indent=2))
    print("\nwrote nnunet/results/logs/oof_calibrate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
