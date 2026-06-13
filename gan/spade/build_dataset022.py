"""Assemble Dataset022_LGEspade = real Dataset000 (SAX/2CH/4CH) + per-fold synthetic (SYNfK_*).
Splits are LEAKAGE-CLEAN per fold: fold k TRAIN = real-fold-k-train + SYNfK_* ONLY (that fold's own synthetic,
from a GAN blind to fold k's test patients + 41-47); VAL = real-fold-k-val (real only, never a synthetic case).

Two modes (synthetic SYNf0..SYNf4 must already be written into Dataset021 by gen_synthetic.py):
  --mode assemble : copy real cases + dataset.json into the raw dataset  (run BEFORE plan_and_preprocess)
  --mode splits   : write the per-fold splits into the preprocessed dir   (run AFTER plan_and_preprocess)
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

ROOT = Path("/home/youssef/projects/research/CMR-MULTI")
SRC = ROOT / "nnunet/nnUNet_raw/Dataset000_LGEgeneralist"
DST = ROOT / "nnunet/nnUNet_raw/Dataset022_LGEspade"
PRE0 = ROOT / "nnunet/nnUNet_preprocessed/Dataset000_LGEgeneralist"
PRE21 = ROOT / "nnunet/nnUNet_preprocessed/Dataset022_LGEspade"


def assemble():
    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)
    n_real = 0
    for img in (SRC / "imagesTr").glob("*_0000.nii.gz"):
        shutil.copy(img, DST / "imagesTr" / img.name)
    for lab in (SRC / "labelsTr").glob("*.nii.gz"):
        shutil.copy(lab, DST / "labelsTr" / lab.name); n_real += 1
    syn = sorted(p.name.replace(".nii.gz", "") for p in (DST / "labelsTr").glob("SYNf*.nii.gz"))
    n_total = len(list((DST / "labelsTr").glob("*.nii.gz")))
    by_fold = {k: len([s for s in syn if s.startswith(f"SYNf{k}_")]) for k in range(5)}
    dj = dict(json.load(open(SRC / "dataset.json")))
    dj["numTraining"] = n_total
    json.dump(dj, open(DST / "dataset.json", "w"), indent=2)
    print(f"[assemble] {n_real} real + {len(syn)} synthetic = {n_total} cases; per-fold synth {by_fold}")


def splits():
    base = json.load(open(PRE0 / "splits_final.json"))
    new = []
    for k, fold in enumerate(base):
        syn_k = sorted(p.name.replace(".nii.gz", "") for p in (DST / "labelsTr").glob(f"SYNf{k}_*.nii.gz"))
        new.append({"train": list(fold["train"]) + syn_k, "val": list(fold["val"])})
        print(f"  fold {k}: train {len(fold['train'])} real + {len(syn_k)} synth = {len(new[k]['train'])} | val {len(fold['val'])} real-only")
    PRE21.mkdir(parents=True, exist_ok=True)
    json.dump(new, open(PRE21 / "splits_final.json", "w"))
    print(f"[splits] wrote {PRE21 / 'splits_final.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["assemble", "splits"], required=True)
    a = ap.parse_args()
    assemble() if a.mode == "assemble" else splits()
