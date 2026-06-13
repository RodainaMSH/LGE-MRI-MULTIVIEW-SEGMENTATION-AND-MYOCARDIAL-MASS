"""Convert LGE_MULTI -> nnU-Net v2 format, mirroring our U-Net pipeline flow.

Creates 5 nnU-Net datasets (the original LGE_MULTI is NOT modified — this only
COPIES + reorganizes files into nnU-Net's required layout):

    Dataset000_LGEgeneralist  pooled SAX+2CH+4CH, 5-class   (= Stage 1 generalist)
    Dataset001_LGESAX         SAX,  5-class                 (= SAX expert)
    Dataset002_LGE2CH         2CH,  5-class (no class-4)     (= 2CH expert)
    Dataset003_LGE4CH         4CH,  5-class                 (= 4CH expert)
    Dataset004_LGERAS         RAS,  2-class                 (= RAS expert)

For each dataset we copy the official TR (train) and VAL (validation) patients
into imagesTr/labelsTr, and stage a splits_final.json with fold 0 =
{train = all *_TR cases, val = all *_VAL cases}. This uses the dataset authors'
official train/val split (the *_VAL folders they added).

Run AFTER this: nnUNetv2_plan_and_preprocess, then inject_splits.py.

"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "LGE_MULTI"
RAW = ROOT / "nnunet" / "nnUNet_raw"
SPLITS_STAGE = ROOT / "nnunet" / "staged_splits"

# Official split: train -> *_TR folders, val -> *_VAL folders.
SPLIT_SUFFIX = {"train": "TR", "val": "VAL"}

# 5-class label map for LV views (matches our scheme). RAS is 2-class.
LV_LABELS = {
    "background": 0,
    "LV_cavity": 1,
    "LV_myocardium": 2,
    "scar": 3,
    "RV_cavity": 4,
}
RAS_LABELS = {"background": 0, "right_atrium": 1}

# (dataset_id, name, views, labels, cohort)
DATASETS = [
    (0, "LGEgeneralist", ["SAX", "2CH", "4CH"], LV_LABELS, "LV"),
    (1, "LGESAX", ["SAX"], LV_LABELS, "LV"),
    (2, "LGE2CH", ["2CH"], LV_LABELS, "LV"),
    (3, "LGE4CH", ["4CH"], LV_LABELS, "LV"),
    (4, "LGERAS", ["RAS"], RAS_LABELS, "RAS"),
]


def convert_one(did: int, name: str, views: list[str], labels: dict, cohort: str):
    ds_name = f"Dataset{did:03d}_{name}"
    ds_dir = RAW / ds_name
    imagesTr = ds_dir / "imagesTr"
    labelsTr = ds_dir / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    # Copy *_TR (train) + *_VAL (val) patients into imagesTr/labelsTr.
    # fold 0 = {train = all TR cases, val = all VAL cases}.
    train_cases, val_cases = [], []
    n_copied = 0
    for view in views:
        for split_kind, suffix in (("train", "TR"), ("val", "VAL")):
            img_dir = DATA / f"{view}_{suffix}" / "image"
            lbl_dir = DATA / f"{view}_{suffix}" / "anno"
            if not img_dir.exists():
                continue
            for src_img in sorted(img_dir.glob(f"LGE_{view}_*.nii.gz")):
                src_lbl = lbl_dir / src_img.name
                if not src_lbl.exists():
                    continue
                pid = int(src_img.stem.split("_")[-1].split(".")[0])
                case = f"{view}_{pid:03d}"
                shutil.copy(src_img, imagesTr / f"{case}_0000.nii.gz")
                shutil.copy(src_lbl, labelsTr / f"{case}.nii.gz")
                n_copied += 1
                (train_cases if split_kind == "train" else val_cases).append(case)

    # dataset.json
    dataset_json = {
        "channel_names": {"0": "LGE_MRI"},
        "labels": labels,
        "numTraining": n_copied,
        "file_ending": ".nii.gz",
        "description": f"{ds_name} from CMR-MULTI LGE_MULTI ({', '.join(views)})",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    # Stage splits (fold 0 = our train/val). Injected into preprocessed/ later.
    SPLITS_STAGE.mkdir(parents=True, exist_ok=True)
    splits = [{"train": sorted(train_cases), "val": sorted(val_cases)}]
    (SPLITS_STAGE / f"{ds_name}.json").write_text(json.dumps(splits, indent=2))

    print(f"  {ds_name}: {n_copied} cases  "
          f"(fold0 train={len(train_cases)}, val={len(val_cases)})")
    return ds_name


def main() -> int:
    if not DATA.exists():
        print(f"FAIL: {DATA} not found.")
        return 2
    print("Converting LGE_MULTI -> nnU-Net format (original NOT modified):\n")
    for (did, name, views, labels, cohort) in DATASETS:
        convert_one(did, name, views, labels, cohort)
    print(f"\nDone. Raw datasets in {RAW.relative_to(ROOT)}")
    print(f"Staged splits in {SPLITS_STAGE.relative_to(ROOT)}")
    print("\nNext: run nnUNetv2_plan_and_preprocess, then inject_splits.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
