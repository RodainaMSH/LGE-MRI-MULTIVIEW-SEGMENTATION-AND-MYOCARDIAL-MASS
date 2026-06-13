"""Unified, apples-to-apples DSC on the official VAL set (the Task-2 metric).

Official Task-2 DSC = Dice averaged over all structures (labels), all views, all
patients. We compute it on the official validation patients (the *_VAL folders:
LV 41-47, RAS 81-94) from each track's per-view NIfTI masks vs the ground-truth
annotations.

Mask sources (only tracks that have val predictions are scored):
    unet_vanilla / unet_efficientnet : {track}/predictions/val/{VIEW}/{VIEW}_{pid:03d}.nii.gz
                                       (produced by their export_masks.py --split val)
    nnunet                           : nnunet/predictions/val/expert_{VIEW}/{VIEW}_{pid:03d}.nii.gz

Per-view foreground labels (Task 2):
    SAX/4CH = {1:LVcav, 2:LVmyo, 3:scar, 4:RVcav}
    2CH     = {1:LVcav, 2:LVmyo, 3:scar}
    RAS     = {1:RightAtrium}

Output: unified_eval/dsc_matrix_val.md + .json

Usage:
    python unified_eval/compute_dsc.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "LGE_MULTI"

VIEW_FG = {"SAX": [1, 2, 3, 4], "2CH": [1, 2, 3], "4CH": [1, 2, 3, 4], "RAS": [1]}
CLASS_NAME = {1: "LVcav/RA", 2: "LVmyo", 3: "scar", 4: "RVcav"}
TRACKS = ("unet_vanilla", "unet_efficientnet", "nnunet")


def dice_np(pred: np.ndarray, gt: np.ndarray, classes) -> dict[int, float]:
    out = {}
    for c in classes:
        p = (pred == c)
        g = (gt == c)
        denom = int(p.sum() + g.sum())
        if denom == 0:
            continue  # class absent in both -> skip (NaN-safe)
        out[c] = float(2 * int((p & g).sum()) / denom)
    return out


def val_pids(view: str) -> list[int]:
    d = DATA / f"{view}_VAL" / "image"
    if not d.exists():
        return []
    return sorted(int(p.stem.split("_")[-1].split(".")[0]) for p in d.glob(f"LGE_{view}_*.nii.gz"))


def gt_path(view: str, pid: int) -> Path:
    return DATA / f"{view}_VAL" / "anno" / f"LGE_{view}_{pid:03d}.nii.gz"


def pred_path(track: str, view: str, pid: int) -> Path:
    if track == "nnunet":
        return ROOT / "nnunet" / "predictions" / "val" / f"expert_{view}" / f"{view}_{pid:03d}.nii.gz"
    return ROOT / track / "predictions" / "val" / view / f"{view}_{pid:03d}.nii.gz"


def evaluate(track: str) -> dict | None:
    per_class: dict[int, list[float]] = defaultdict(list)
    per_view: dict[str, list[float]] = defaultdict(list)
    all_dice: list[float] = []
    found = 0
    for view in ("SAX", "2CH", "4CH", "RAS"):
        for pid in val_pids(view):
            gp, pp = gt_path(view, pid), pred_path(track, view, pid)
            if not gp.exists() or not pp.exists():
                continue
            found += 1
            gt = np.asarray(nib.load(str(gp)).dataobj).astype(int)
            pr = np.asarray(nib.load(str(pp)).dataobj).astype(int)
            if pr.shape != gt.shape:
                print(f"  [{track}] shape mismatch {view}_{pid}: pred{pr.shape} vs gt{gt.shape} -> skip")
                continue
            for c, val in dice_np(pr, gt, VIEW_FG[view]).items():
                per_class[c].append(val)
                per_view[view].append(val)
                all_dice.append(val)
    if found == 0:
        return None  # track not evaluated on val yet
    return {
        "track": track,
        "dsc_overall": float(np.mean(all_dice)) if all_dice else float("nan"),
        "n_evals": len(all_dice),
        "per_view": {v: (float(np.mean(x)) if x else float("nan")) for v, x in per_view.items()},
        "per_class": {CLASS_NAME[c]: (float(np.mean(x)) if x else float("nan"))
                      for c, x in sorted(per_class.items())},
    }


def main() -> None:
    rows = [r for r in (evaluate(t) for t in TRACKS) if r is not None]
    if not rows:
        print("No track has val predictions yet. Run export_masks.py --split val first.")
        return

    lines = ["# Unified official DSC on the VAL set (LV 41-47, RAS 81-94)\n"]
    lines.append("DSC = Dice averaged over all labels, views, patients — the 70% "
                 "term of the Task 2 score. Native-geometry NIfTI vs ground truth.\n")
    lines.append("| Track | DSC (overall) | SAX | 2CH | 4CH | RAS | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        pv = r["per_view"]
        lines.append(f"| `{r['track']}` | **{r['dsc_overall']:.4f}** | "
                     f"{pv.get('SAX', float('nan')):.3f} | {pv.get('2CH', float('nan')):.3f} | "
                     f"{pv.get('4CH', float('nan')):.3f} | {pv.get('RAS', float('nan')):.3f} | "
                     f"{r['n_evals']} |")
    lines.append("\n## Per-class DSC\n")
    lines.append("| Track | LVcav/RA | LVmyo | scar | RVcav |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        pc = r["per_class"]
        lines.append(f"| `{r['track']}` | {pc.get('LVcav/RA', float('nan')):.3f} | "
                     f"{pc.get('LVmyo', float('nan')):.3f} | {pc.get('scar', float('nan')):.3f} | "
                     f"{pc.get('RVcav', float('nan')):.3f} |")
    md = "\n".join(lines) + "\n"

    (ROOT / "unified_eval" / "dsc_matrix_val.md").write_text(md)
    (ROOT / "unified_eval" / "dsc_matrix_val.json").write_text(json.dumps(rows, indent=2))
    print(md)
    print("Wrote unified_eval/dsc_matrix_val.md + .json")


if __name__ == "__main__":
    main()
