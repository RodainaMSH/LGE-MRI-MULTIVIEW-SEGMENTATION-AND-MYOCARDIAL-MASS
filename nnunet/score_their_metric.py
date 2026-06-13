"""Score any nnU-Net run's OOF predictions under the CHALLENGE BASELINE metric
(qutaiping/CMR_multi_baseline 2D_seg/metrics.py dice_score) — the metric Youssef wants us to
report under from now on. Replicates their test.py evaluate_dataset: per view, all slices,
batch_size=4, per-class dice (smooth=1e-6 -> empty-in-both = 1.0), mean over batches; mean_dice
over fg classes per view; overall = mean of per-view fg-means (their test.py 'Overall').

Usage: PYTHONPATH=. python nnunet/score_their_metric.py <run_dirname> [<run_dirname> ...]
  e.g. nnUNetTrainerSMP_effv2xl__nnUNetPlans__2d
Pass 'ALL' to score every full-5-fold-OOF run.
"""
from __future__ import annotations
import sys, os, glob
from pathlib import Path
import numpy as np, torch, SimpleITK as sitk
sys.path.insert(0, "/tmp/theirmetric")
from metrics import dice_score, mean_dice   # THEIR exact code

ROOT = Path("/home/youssef/projects/research/CMR-MULTI")
RESB = ROOT / "nnunet/nnUNet_results/Dataset000_LGEgeneralist"
# their config num_classes per view (bg + structures): sa/4ch=5, 2ch=4, ras154=2
VIEW_NC = {"SAX": 5, "4CH": 5, "2CH": 4, "RAS": 2}
SCAR = 3
BS = 4  # their test.py default batch_size

def view_dice(run, view, nc):
    """all-slice, batch_size=4, their dice_score; returns per-class list + n_slices."""
    cmap = {}
    for k in range(5):
        for p in (RESB / run / f"fold_{k}" / "validation").glob(f"{view}_*.nii.gz"):
            cmap[int(p.name.replace(".nii.gz", "").split("_")[1])] = p
    if not cmap:
        return None, 0
    sl_p, sl_g = [], []
    for pid in sorted(cmap):
        gtf = ROOT / "LGE_MULTI" / f"{view}_TR" / "anno" / f"LGE_{view}_{pid:03d}.nii.gz"
        if not gtf.exists():
            continue
        pr = sitk.GetArrayFromImage(sitk.ReadImage(str(cmap[pid]))).astype(np.int64)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gtf))).astype(np.int64)
        if pr.shape != gt.shape:
            continue
        for d in range(gt.shape[0]):
            sl_p.append(pr[d]); sl_g.append(gt[d])
    if not sl_p:
        return None, 0
    P = torch.from_numpy(np.stack(sl_p)); G = torch.from_numpy(np.stack(sl_g))
    acc = [0.0] * nc; nb = 0
    for i in range(0, P.shape[0], BS):
        dc = dice_score(P[i:i+BS], G[i:i+BS], nc)
        for c in range(nc): acc[c] += dc[c]
        nb += 1
    return [v / nb for v in acc], P.shape[0]

def score_run(run):
    print(f"\n=== {run.replace('__nnUNetPlans__2d','')} (THEIR metric, bs={BS}, all-slice OOF) ===")
    view_fg_means, scar_vals = [], []
    for view, nc in VIEW_NC.items():
        per, n = view_dice(run, view, nc)
        if per is None:
            print(f"  {view}: (no preds)"); continue
        fg = mean_dice(per)  # mean of classes 1..nc-1
        view_fg_means.append(fg)
        scar = per[SCAR] if nc > SCAR else None
        if scar is not None: scar_vals.append(scar)
        cls = " ".join(f"c{c}:{per[c]:.3f}" for c in range(1, nc))
        print(f"  {view:>3} ({n:>3} sl): fg-mean {fg:.4f} | scar {scar if scar is None else f'{scar:.4f}'} | {cls}")
    overall = float(np.mean(view_fg_means)) if view_fg_means else float("nan")
    scar_mean = float(np.mean(scar_vals)) if scar_vals else float("nan")
    print(f"  >>> OVERALL (mean of per-view fg-means) = {overall:.4f} | SCAR (mean over SAX/2CH/4CH) = {scar_mean:.4f}")
    return overall, scar_mean

def main():
    args = sys.argv[1:]
    if not args or args[0] == "ALL":
        runs = []
        for d in sorted(RESB.glob("*__nnUNetPlans__2d")):
            if all((d / f"fold_{k}" / "validation").glob("*.nii.gz") and
                   len(list((d / f"fold_{k}" / "validation").glob("*.nii.gz"))) > 0 for k in range(5)):
                runs.append(d.name)
    else:
        runs = [a if a.endswith("2d") else f"{a}__nnUNetPlans__2d" for a in args]
    results = {}
    for run in runs:
        try:
            results[run] = score_run(run)
        except Exception as e:
            print(f"  {run}: ERROR {e}")
    print("\n================ SUMMARY (their metric, OOF) ================")
    for run, (ov, sc) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"  {run.replace('__nnUNetPlans__2d',''):<34} overall {ov:.4f}  scar {sc:.4f}")

if __name__ == "__main__":
    main()
