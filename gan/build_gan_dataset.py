"""Assemble Dataset020_LGEgan = real Dataset000 (SAX/2CH/4CH) + the synthetic SAXSYN cases.
Splits: each fold's TRAIN = real-fold-train + ALL synthetic; VAL = real-fold-val only (synthetic never validated).
So OOF is scored on real held-out patients; augmentation only enters training. (Feasibility = mild leakage since
the all-patients GAN saw val patients; per-fold GANs remove that next.)

Usage: PYTHONPATH=. python gan/build_gan_dataset.py
"""
from __future__ import annotations
import json, shutil
from pathlib import Path

ROOT = Path("/home/youssef/projects/research/CMR-MULTI")
SRC = ROOT / "nnunet/nnUNet_raw/Dataset000_LGEgeneralist"
DST = ROOT / "nnunet/nnUNet_raw/Dataset020_LGEgan"
PRE0 = ROOT / "nnunet/nnUNet_preprocessed/Dataset000_LGEgeneralist"
PRE20 = ROOT / "nnunet/nnUNet_preprocessed/Dataset020_LGEgan"

def main():
    # 1) copy real cases in (synthetic SAXSYN_* already present from gen_synthetic.py)
    n_real = 0
    for img in (SRC / "imagesTr").glob("*_0000.nii.gz"):
        shutil.copy(img, DST / "imagesTr" / img.name)
    for lab in (SRC / "labelsTr").glob("*.nii.gz"):
        shutil.copy(lab, DST / "labelsTr" / lab.name); n_real += 1
    syn = sorted(p.name.replace(".nii.gz", "") for p in (DST / "labelsTr").glob("SAXSYN_*.nii.gz"))
    n_total = len(list((DST / "labelsTr").glob("*.nii.gz")))
    # 2) dataset.json
    dj = dict(json.load(open(SRC / "dataset.json")))
    dj["numTraining"] = n_total
    json.dump(dj, open(DST / "dataset.json", "w"), indent=2)
    # 3) splits: synthetic into every fold's train
    base = json.load(open(PRE0 / "splits_final.json"))
    new = []
    for fold in base:
        new.append({"train": list(fold["train"]) + syn, "val": list(fold["val"])})
    PRE20.mkdir(parents=True, exist_ok=True)
    json.dump(new, open(PRE20 / "splits_final.json", "w"))
    print(f"Dataset020_LGEgan: {n_real} real + {len(syn)} synthetic = {n_total} cases")
    print(f"  fold0 train {len(new[0]['train'])} ({len(syn)} synth) / val {len(new[0]['val'])} (real only)")
    print(f"  splits written to {PRE20/'splits_final.json'} (copy AFTER plan_and_preprocess if it overwrites)")

if __name__ == "__main__":
    main()
