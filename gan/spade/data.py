"""Data adapter — our nnUNet_raw LGE slices -> the dict SpadeModel.preprocess_input expects.

Reuses the SAME data source + leakage-clean fold exclusion that Youssef vetted in gan/train_pix2pix.py:
  excl = (all pids 1-47) - (fold-k TRAIN pids)  =  fold-k's 8 test pids + the 7-val (41-47).
So a per-fold GAN sees ONLY that fold's 32 training patients. Identical rule to v1 — do not change.

Output per item: {'label': LongTensor[1,128,128] in {0..4}, 'image': FloatTensor[1,128,128] in [-1,1]}.
Normalization: per-slice robust min-max (1/99 pct) -> [-1,1]. Their loader fed pre-'normalized' images
to a tanh generator; this is our explicit, contrast-preserving choice (scar is the bright tail -> stays
near +1, myo mid). FLAGGED gap #2 — chosen deliberately, revisit if synthesis washes out.
"""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path("/home/youssef/projects/research/CMR-MULTI")
RAW = ROOT / "nnunet/nnUNet_raw/Dataset000_LGEgeneralist"
SPLITS = ROOT / "nnunet/nnUNet_preprocessed/Dataset000_LGEgeneralist/splits_final.json"
NCLASS = 5
SIZE = 128


def fold_exclude(exclude_fold):
    """Train-only exclusion set for a fold (=8 fold-test pids + 41-47). exclude_fold<0 -> none."""
    if exclude_fold is None or exclude_fold < 0:
        return set()
    splits = json.load(open(SPLITS))
    train_pids = {int(c.split("_")[1]) for c in splits[exclude_fold]["train"]}
    all_pids = {int(p.name.replace(".nii.gz", "").split("_")[1]) for p in (RAW / "labelsTr").glob("SAX_*.nii.gz")}
    return all_pids - train_pids


def _resize(a, order):
    t = torch.from_numpy(a.astype(np.float32))[None, None]
    if order == 1:
        return F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()
    return F.interpolate(t, size=(SIZE, SIZE), mode="nearest")[0, 0].numpy()


def norm_img(a):
    """Per-slice robust min-max -> [-1,1] (contrast-preserving)."""
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = np.clip(a, lo, hi)
    return 2 * (a - lo) / max(hi - lo, 1e-6) - 1


def load_fold_slices(views=("SAX", "2CH", "4CH"), exclude_pids=(), scar_only=False):
    """Return list of (view, pid, z, image[H,W] raw, label[H,W] int) for the kept patients/views.
    Used by both training (all slices) and generation (scar_only=True -> only scar-bearing slices)."""
    exclude_pids = set(exclude_pids)
    out = []
    for lab in sorted((RAW / "labelsTr").glob("*.nii.gz")):
        stem = lab.name.replace(".nii.gz", "")
        v, pid = stem.split("_")[0], int(stem.split("_")[1])
        if v not in views or pid in exclude_pids:
            continue
        img = RAW / "imagesTr" / f"{v}_{pid:03d}_0000.nii.gz"
        if not img.exists():
            continue
        g = np.asarray(nib.load(str(lab)).dataobj).astype(np.int64)
        a = np.asarray(nib.load(str(img)).dataobj).astype(np.float32)
        for z in range(a.shape[2]):
            gz = g[:, :, z]
            if scar_only and (gz == 3).sum() < 20:
                continue
            out.append((v, pid, z, a[:, :, z].copy(), gz.copy()))
    return out


class SpadeSlices(Dataset):
    """All slices from the chosen views for a fold's train patients (label + [-1,1] image, 128x128)."""
    def __init__(self, views=("SAX", "2CH", "4CH"), exclude_pids=()):
        self.items = load_fold_slices(views, exclude_pids, scar_only=False)
        n_scar = sum(1 for *_, g in self.items if (g == 3).sum() >= 20)
        print(f"[SPADE data] {len(self.items)} slices ({n_scar} scar) from views={views}, excluded {len(set(exclude_pids))} pids")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        _, _, _, a, g = self.items[i]
        img = _resize(norm_img(a), 1)[None]                 # 1 x 128 x 128, [-1,1]
        lab = _resize(g.astype(np.float32), 0)[None]        # 1 x 128 x 128, {0..4}
        return {"label": torch.from_numpy(lab).long(), "image": torch.from_numpy(img).float()}
