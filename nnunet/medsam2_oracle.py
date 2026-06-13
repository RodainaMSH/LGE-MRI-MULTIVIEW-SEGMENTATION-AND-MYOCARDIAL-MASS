"""ORACLE test: does off-the-shelf MedSAM2 reach the paper's scar DSC (~65.37) when given
GROUND-TRUTH prompts? This is the strongest faithful version of the paper's (undocumented but
inferred) protocol: per-structure GT-derived prompts on a class-agnostic model.

For each SAX patient, every slice with GT scar, every connected component of GT scar:
  derive a GT box (and centroid) -> prompt full MedSAM2 (2D image predictor, off-the-shelf,
  NO fine-tune) -> union refined masks -> scar DSC vs GT.

Prompt variants (run several to bracket the oracle ceiling, since exact protocol is unknown):
  box        : GT bounding box per component  (the standard MedSAM2 box benchmark)
  box_point  : GT box + positive centroid point
  point      : positive centroid point only

This is an UPPER BOUND (uses GT at test time) -> NOT deployable, purely diagnostic:
  - if oracle scar DSC ~>= 0.65  -> the paper's 65.37 is real but PROMPT-DEPENDENT (unreachable
    by our fully-automatic cascade, best 0.481). Conclusion confirmed.
  - if oracle scar DSC <<  0.65  -> even perfect prompts don't get there on OUR data; the gap is
    cohort/data, not just prompting.

Usage:
  PYTHONPATH=specialists/MedSAM2 python nnunet/medsam2_oracle.py --prompt box [--limit N]
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path("/home/youssef/projects/research/CMR-MULTI")
REPO = ROOT / "specialists" / "MedSAM2"
SCAR = 3
PAD = 4

def norm_uint8(vol):
    lo, hi = np.percentile(vol, [0.5, 99.5])
    if hi <= lo: hi, lo = vol.max(), vol.min()
    if hi <= lo: return np.zeros_like(vol, np.uint8)
    return ((np.clip(vol, lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)

def components(mask2d, H, W):
    """per-connected-component: (box[x0,y0,x1,y1], centroid[x,y])."""
    lab, n = ndi.label(mask2d)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        if xs.size < 2:
            continue
        x0, x1 = max(0, xs.min() - PAD), min(W - 1, xs.max() + PAD)
        y0, y1 = max(0, ys.min() - PAD), min(H - 1, ys.max() + PAD)
        cx, cy = float(xs.mean()), float(ys.mean())
        out.append(([x0, y0, x1, y1], [cx, cy]))
    return out

def dice(p, g):
    p, g = p.astype(bool), g.astype(bool)
    d = int(p.sum() + g.sum())
    return None if d == 0 else 2.0 * int((p & g).sum()) / d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="SAX")
    ap.add_argument("--prompt", choices=["box", "box_point", "point"], default="box")
    ap.add_argument("--cfg", default="configs/sam2.1_hiera_t512.yaml")
    ap.add_argument("--ckpt", default="checkpoints/MedSAM2_latest.pt")
    ap.add_argument("--limit", type=int, default=0, help="cap #patients (0=all)")
    ap.add_argument("--select", choices=["iou", "gt"], default="iou",
                    help="pick among MedSAM2's 3 masks by predicted-IoU (deployable) or by true DSC vs GT (absolute ceiling)")
    ap.add_argument("--box_from", choices=["gt", "coarse"], default="gt",
                    help="box source: gt (oracle) or coarse (automatic, from nnU-Net OOF pred) -- matched cohort comparison")
    ap.add_argument("--coarse", default="nnunet/nnUNet_results/Dataset000_LGEgeneralist/nnUNetTrainerSMP_effv2xl__nnUNetPlans__2d",
                    help="coarse run dir (OOF preds) for --box_from coarse")
    args = ap.parse_args()

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    os.chdir(REPO)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(args.cfg, args.ckpt, device=dev)
    predictor = SAM2ImagePredictor(model)

    anno_dir = ROOT / "LGE_MULTI" / f"{args.view}_TR" / "anno"
    img_dir = ROOT / "LGE_MULTI" / f"{args.view}_TR" / "image"
    pids = sorted(int(p.name.replace(".nii.gz", "").split("_")[-1]) for p in anno_dir.glob(f"LGE_{args.view}_*.nii.gz"))
    if args.limit:
        pids = pids[:args.limit]
    # for --box_from coarse: map each pid -> its OOF coarse prediction (the fold where it was in val)
    coarse_map = {}
    if args.box_from == "coarse":
        for k in range(5):
            for p in (ROOT / args.coarse / f"fold_{k}" / "validation").glob(f"{args.view}_*.nii.gz"):
                coarse_map[int(p.name.replace(".nii.gz", "").split("_")[1])] = p
    print(f"ORACLE [{args.prompt}/select={args.select}/box_from={args.box_from}] off-the-shelf MedSAM2 | "
          f"{args.view} | {len(pids)} patients")

    per_pt, n_box, n_skip_noscar = [], 0, 0
    for pid in pids:
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_dir / f"LGE_{args.view}_{pid:03d}.nii.gz"))).astype(np.float32)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(anno_dir / f"LGE_{args.view}_{pid:03d}.nii.gz"))).astype(int)
        if (gt == SCAR).sum() == 0:
            n_skip_noscar += 1; continue
        # box source volume: GT (oracle) or coarse nnU-Net pred (automatic). Cohort = GT-scar patients either way.
        if args.box_from == "coarse":
            if pid not in coarse_map:
                print(f"  {args.view}_{pid:03d}: no coarse pred -> DSC 0"); per_pt.append(0.0); continue
            box_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(coarse_map[pid]))).astype(int) == SCAR
        else:
            box_vol = gt == SCAR
        u8 = norm_uint8(img); D, H, W = img.shape
        refined = np.zeros_like(gt, np.uint8)
        for d in range(D):
            comps = components(box_vol[d], H, W)
            if not comps:
                continue
            rgb = np.repeat(u8[d][:, :, None], 3, axis=2)
            predictor.set_image(rgb)
            for box, cen in comps:
                n_box += 1
                kw = {}
                if args.prompt in ("box", "box_point"):
                    kw["box"] = np.array(box)[None, :]
                if args.prompt in ("point", "box_point"):
                    kw["point_coords"] = np.array([cen]); kw["point_labels"] = np.array([1])
                with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16):
                    m, sc, _ = predictor.predict(multimask_output=True, **kw)
                if args.select == "gt":  # absolute ceiling: pick the mask with best true DSC vs GT scar on this slice
                    gd = gt[d] == SCAR
                    best, bi = -1.0, 0
                    for i in range(m.shape[0]):
                        dd = dice(m[i] > 0, gd) or 0.0
                        if dd > best: best, bi = dd, i
                    m = m[bi]
                else:
                    m = m[int(np.argmax(sc))]  # deployable: best of 3 by predicted IoU
                refined[d][m > 0] = SCAR
        dk = dice(refined == SCAR, gt == SCAR)
        if dk is not None:
            per_pt.append(dk)
        print(f"  {args.view}_{pid:03d}: oracle_scar_DSC={dk:.3f}  (gt_scar_vox={(gt==SCAR).sum()})", flush=True)

    arr = np.array(per_pt)
    print(f"\n==== ORACLE RESULT [{args.prompt}] ====")
    print(f"  patients with scar: {len(per_pt)}  (skipped {n_skip_noscar} no-scar)   boxes: {n_box}")
    print(f"  ORACLE scar DSC = {arr.mean():.4f} +/- {arr.std():.4f}  (median {np.median(arr):.4f})")
    print(f"  [paper MedSAM2 scar (Table D5) = 0.6537 | our AUTO cascade best = 0.481 | V2-XL auto = 0.4112]")

if __name__ == "__main__":
    main()
