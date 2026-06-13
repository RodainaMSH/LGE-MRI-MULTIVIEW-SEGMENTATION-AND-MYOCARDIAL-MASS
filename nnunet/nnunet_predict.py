"""Generalist-only nnU-Net inference (rebuilt, replaces the deleted MoE predict).

Predicts segmentation masks for a split using the PLAIN GENERALIST for the LV
views (Dataset000, which experts did not beat) and the RAS model for RAS. No
per-view expert models, no fusion here — this only produces masks. Per-view scar
MASS + the multi-view combine happen in nnunet_mass.py.

Masks are written to:  nnunet/predictions/{split}/gen_{VIEW}/{VIEW}_{pid:03d}.nii.gz

Heavy (loads the network; ~1-3 min torch startup per dataset). Run in your terminal:
    source nnunet/env.sh
    python nnunet/nnunet_predict.py --split val 2>&1 | tee nnunet/results/logs/predict_val.log

Then score masks (DSC) with unified_eval/compute_dsc.py (point it at gen_{VIEW}) and
compute scar mass with nnunet/nnunet_mass.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "LGE_MULTI"
PRED = ROOT / "nnunet" / "predictions"

# nnU-Net env (so the script is self-contained even without sourcing env.sh).
os.environ.setdefault("nnUNet_raw", str(ROOT / "nnunet" / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(ROOT / "nnunet" / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(ROOT / "nnunet" / "nnUNet_results"))

SPLIT_SUFFIX = {"train": "TR", "val": "VAL"}
# view -> (dataset_id, trainer). LV views use the generalist (D000, 100ep); RAS uses D004 (70ep).
VIEW_MODEL = {
    "SAX": (0, "nnUNetTrainer_100epochs"),
    "2CH": (0, "nnUNetTrainer_100epochs"),
    "4CH": (0, "nnUNetTrainer_100epochs"),
    "RAS": (4, "nnUNetTrainer_70epochs"),
}


def split_pids(view: str, split: str) -> list[int]:
    d = DATA / f"{view}_{SPLIT_SUFFIX[split]}" / "image"
    if not d.exists():
        return []
    return sorted(int(p.name.split("_")[-1].split(".")[0]) for p in d.glob(f"LGE_{view}_*.nii.gz"))


def predict_view(view: str, split: str, fold: str) -> int:
    did, trainer = VIEW_MODEL[view]
    pids = split_pids(view, split)
    if not pids:
        print(f"  {view}: no {split} images, skipping")
        return 0
    out_dir = PRED / split / f"gen_{view}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = SPLIT_SUFFIX[split]
    with tempfile.TemporaryDirectory() as tmp:
        tin = Path(tmp) / "in"
        tin.mkdir()
        for pid in pids:
            src = DATA / f"{view}_{suffix}" / "image" / f"LGE_{view}_{pid:03d}.nii.gz"
            shutil.copy(src, tin / f"{view}_{pid:03d}_0000.nii.gz")
        cmd = ["nnUNetv2_predict", "-i", str(tin), "-o", str(out_dir),
               "-d", str(did), "-c", "2d", "-f", fold, "-tr", trainer]
        print(f"  {view}: predicting {len(pids)} cases with D{did:03d}/{trainer} fold {fold}")
        subprocess.run(cmd, check=True)
    # nnU-Net writes {VIEW}_{pid}.nii.gz already; clean its bookkeeping files.
    for junk in ("dataset.json", "plans.json", "predict_from_raw_data_args.json"):
        (out_dir / junk).unlink(missing_ok=True)
    return len(pids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--views", nargs="*", default=["SAX", "2CH", "4CH", "RAS"])
    ap.add_argument("--fold", default="0", help="nnU-Net fold (0); use e.g. '0 1 2 3 4' for the ensemble later")
    args = ap.parse_args()

    print(f"Generalist inference on the {args.split} split -> {PRED / args.split}/gen_<VIEW>/\n")
    total = 0
    for view in args.views:
        total += predict_view(view, args.split, args.fold)
    print(f"\nDone: {total} masks written. Next: python nnunet/nnunet_mass.py --split {args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
