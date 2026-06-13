"""Build a 5-fold CV split for the generalist (Dataset000), over the 40 LV TRAIN
patients (IDs 1-40), stratified by scar presence. The 7 official VAL patients
(41-47) are EXCLUDED from every fold so they stay a clean final holdout.

Purpose: out-of-fold (OOF) predictions for all 40 train patients -> honest scar-mass
calibration + ensemble + a CV harness to test future levers without touching the 7 val.

Each LV patient pid has 3 nnU-Net cases: SAX_{pid}, 2CH_{pid}, 4CH_{pid}; all 3 go to
the same fold (patient-grouped -> no view leakage between train/val within a fold).

Deterministic (no RNG): patients sorted, round-robin assigned within each scar stratum.

Backs up the existing splits_final.json before overwriting. Run:
    PYTHONPATH=. python nnunet/make_cv_splits.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "nnunet" / "nnUNet_preprocessed" / "Dataset000_LGEgeneralist"
SPLITS = PRE / "splits_final.json"
VIEWS = ("SAX", "2CH", "4CH")
N_FOLDS = 5
TRAIN_PIDS = list(range(1, 41))   # LV train patients; 41-47 are the held-out official val


def scar_presence() -> dict[int, bool]:
    df = pd.read_excel(ROOT / "LGE_MULTI" / "dataset_lge.xlsx", sheet_name="SAX")
    sq = dict(zip(df["编号"].astype(int), df["Scar_Quality"].astype(float)))
    return {pid: (sq.get(pid, 0.0) > 1e-6) for pid in TRAIN_PIDS}


def assign_folds() -> dict[int, int]:
    """Stratified round-robin: balance scar / no-scar patients across the 5 folds."""
    pres = scar_presence()
    nonzero = sorted(p for p in TRAIN_PIDS if pres[p])
    zero = sorted(p for p in TRAIN_PIDS if not pres[p])
    fold_of = {}
    for i, p in enumerate(nonzero):
        fold_of[p] = i % N_FOLDS
    for i, p in enumerate(zero):
        fold_of[p] = i % N_FOLDS
    return fold_of


def cases(pids) -> list[str]:
    return sorted(f"{v}_{p:03d}" for p in pids for v in VIEWS)


def main() -> int:
    if not SPLITS.exists():
        print(f"FAIL: {SPLITS} not found (run convert/preprocess first).")
        return 2
    fold_of = assign_folds()
    pres = scar_presence()

    splits = []
    for k in range(N_FOLDS):
        val_pids = [p for p in TRAIN_PIDS if fold_of[p] == k]
        train_pids = [p for p in TRAIN_PIDS if fold_of[p] != k]
        splits.append({"train": cases(train_pids), "val": cases(val_pids)})

    # ---- sanity checks (fail loudly) ----
    all_val = [p for k in range(N_FOLDS) for p in TRAIN_PIDS if fold_of[p] == k]
    assert sorted(all_val) == TRAIN_PIDS, "every train patient must be val exactly once"
    for k, s in enumerate(splits):
        tr, va = set(s["train"]), set(s["val"])
        assert tr.isdisjoint(va), f"fold {k}: train/val overlap"
        bad = [c for c in tr | va if int(c.split("_")[1]) > 40]
        assert not bad, f"fold {k}: leaked holdout patients {bad}"

    print(f"5-fold CV over {len(TRAIN_PIDS)} train patients (41-47 excluded as holdout):")
    for k, s in enumerate(splits):
        vp = sorted({int(c.split("_")[1]) for c in s["val"]})
        nz = sum(pres[p] for p in vp)
        print(f"  fold {k}: val={len(vp)} pts ({nz} scar / {len(vp)-nz} no-scar) "
              f"{vp} | train={len(s['train'])//3} pts")

    backup = SPLITS.with_name("splits_final_officialval.json")
    if not backup.exists():
        shutil.copy(SPLITS, backup)
        print(f"\nbacked up old split -> {backup.name}")
    else:
        print(f"\n(backup {backup.name} already exists, not overwriting)")
    SPLITS.write_text(json.dumps(splits, indent=2))
    print(f"wrote 5-fold split -> {SPLITS}")
    print("\nNEXT: rename the official-split fold_0 model, then train folds 0-4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
