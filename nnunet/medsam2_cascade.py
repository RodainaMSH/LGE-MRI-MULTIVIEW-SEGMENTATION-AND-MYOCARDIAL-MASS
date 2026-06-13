"""Auto-cascade scar refinement with the FULL MedSAM2 model (zero-shot, no fine-tune yet).

Pipeline (fully automatic, comparable to our nnU-Net OOF numbers):
  nnU-Net coarse OOF prediction  ->  per-slice, per-connected-component box around coarse scar
  ->  MedSAM2 2D image predictor (box prompt)  ->  refined scar mask  ->  dice_np scar DSC.

Compares, on the SAME fold-val SAX patients:
  (a) COARSE scar DSC (the nnU-Net coarse mask itself)
  (b) CASCADE scar DSC (MedSAM2-refined)
The delta is the decision: does the full MedSAM2 pipeline improve scar over coarse?

Honest caveat baked in: boxes come from coarse SCAR, so the cascade can only refine scar
the coarse model already found -- it cannot recover false-negative scar. This measures the
refinement ceiling of the zero-shot full pipeline before we invest a night in fine-tuning.

Usage:
  PYTHONPATH=specialists/MedSAM2 python nnunet/medsam2_cascade.py \
      --fold 0 --view SAX \
      --coarse nnunet/nnUNet_results/Dataset000_LGEgeneralist/nnUNetTrainerSMP_effv2xl__nnUNetPlans__2d \
      [--out nnunet/predictions/cascade/medsam2_zs_f0]
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path("/home/youssef/projects/research/CMR-MULTI")
SCAR = 3
PAD = 4  # bbox padding (px), like their get_bbox bbox_shift

def norm_uint8(vol: np.ndarray) -> np.ndarray:
    """Robust per-volume percentile normalization to [0,255]."""
    lo, hi = np.percentile(vol, [0.5, 99.5])
    if hi <= lo:
        hi = vol.max(); lo = vol.min()
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.uint8)
    v = np.clip(vol, lo, hi)
    v = (v - lo) / (hi - lo) * 255.0
    return v.astype(np.uint8)

def comp_boxes(mask2d: np.ndarray, H: int, W: int):
    """One box per connected component of a 2D binary mask. box = [x0,y0,x1,y1]."""
    lab, n = ndi.label(mask2d)
    boxes = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        if xs.size < 2:  # skip 1-px noise
            continue
        x0 = max(0, xs.min() - PAD); x1 = min(W - 1, xs.max() + PAD)
        y0 = max(0, ys.min() - PAD); y1 = min(H - 1, ys.max() + PAD)
        boxes.append([x0, y0, x1, y1])
    return boxes

def dice(p: np.ndarray, g: np.ndarray):
    p = p.astype(bool); g = g.astype(bool)
    denom = int(p.sum() + g.sum())
    if denom == 0:
        return None  # absent in both -> skip (matches dice_np)
    return 2.0 * int((p & g).sum()) / denom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--view", default="SAX")
    ap.add_argument("--coarse", required=True, help="nnU-Net run dir (has fold_k/validation/*.nii.gz)")
    ap.add_argument("--cfg", default="configs/sam2.1_hiera_t512.yaml")
    ap.add_argument("--ckpt", default="checkpoints/MedSAM2_latest.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--multimask", action="store_true", help="pick best of 3 masks by score")
    ap.add_argument("--ckpt_ft", default=None, help="fine-tuned prompt+mask-decoder state dict")
    ap.add_argument("--box_class", type=int, default=SCAR, help="coarse class to box (3=scar, 2=myo)")
    args = ap.parse_args()
    BOX = args.box_class

    view = args.view
    cdir = ROOT / args.coarse / f"fold_{args.fold}" / "validation"
    preds = sorted(cdir.glob(f"{view}_*.nii.gz"))
    assert preds, f"no {view} preds in {cdir}"
    print(f"fold {args.fold} {view}: {len(preds)} patients  coarse={args.coarse.split('/')[-1]}")

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    os.chdir(ROOT / "specialists" / "MedSAM2")  # cfg/ckpt are relative to repo
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(args.cfg, args.ckpt, device=dev)
    tag_ft = "zero-shot"
    if args.ckpt_ft:
        ft = torch.load(args.ckpt_ft, map_location=dev, weights_only=False)
        model.sam_prompt_encoder.load_state_dict(ft["prompt_encoder"])
        model.sam_mask_decoder.load_state_dict(ft["mask_decoder"])
        tag_ft = f"fine-tuned({ft.get('box_mode','?')}) {Path(args.ckpt_ft).name}"
    print(f"  model: {tag_ft}  | box source = coarse class {BOX}")
    predictor = SAM2ImagePredictor(model)

    out = Path(args.out) if args.out else None
    if out: out.mkdir(parents=True, exist_ok=True)

    coarse_d, casc_d = [], []
    n_box = 0
    for pp in preds:
        pid = int(pp.name.replace(".nii.gz", "").split("_")[1])
        img_f = ROOT / "LGE_MULTI" / f"{view}_TR" / "image" / f"LGE_{view}_{pid:03d}.nii.gz"
        gt_f = ROOT / "LGE_MULTI" / f"{view}_TR" / "anno" / f"LGE_{view}_{pid:03d}.nii.gz"
        if not img_f.exists() or not gt_f.exists():
            print(f"  {view}_{pid:03d}: MISSING img/gt -> skip"); continue
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_f))).astype(np.float32)  # [D,H,W]
        pr = sitk.GetArrayFromImage(sitk.ReadImage(str(pp))).astype(int)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_f))).astype(int)
        if not (img.shape == pr.shape == gt.shape):
            print(f"  {view}_{pid:03d}: shape mismatch {img.shape}/{pr.shape}/{gt.shape} -> skip"); continue
        D, H, W = img.shape
        u8 = norm_uint8(img)
        refined = np.zeros_like(pr, dtype=np.uint8)
        coarse_scar = (pr == SCAR)       # for the COARSE-baseline DSC (always scar)
        box_src = (pr == BOX)            # box source (scar or myo)
        for d in range(D):
            boxes = comp_boxes(box_src[d], H, W)
            if not boxes:
                continue
            rgb = np.repeat(u8[d][:, :, None], 3, axis=2)  # HxWx3 uint8
            predictor.set_image(rgb)
            for bx in boxes:
                n_box += 1
                with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16):
                    m, sc, _ = predictor.predict(box=np.array(bx)[None, :], multimask_output=args.multimask)
                if args.multimask:
                    m = m[int(np.argmax(sc))]
                else:
                    m = m[0]
                refined[d][m > 0] = SCAR
        # DSC scar on this patient (coarse vs cascade)
        dc = dice(coarse_scar, gt == SCAR)
        dk = dice(refined == SCAR, gt == SCAR)
        if dc is not None: coarse_d.append(dc)
        if dk is not None: casc_d.append(dk)
        print(f"  {view}_{pid:03d}: coarse_scar={'n/a' if dc is None else f'{dc:.3f}'}  "
              f"cascade_scar={'n/a' if dk is None else f'{dk:.3f}'}  (gt_scar_vox={(gt==SCAR).sum()})")
        if out:
            o = sitk.GetImageFromArray(refined); o.CopyInformation(sitk.ReadImage(str(pp)))
            sitk.WriteImage(o, str(out / pp.name))

    print("\n==== ZERO-SHOT CASCADE RESULT (fold {}, {}) ====".format(args.fold, view))
    print(f"  boxes prompted: {n_box}")
    print(f"  COARSE  scar DSC (mean over {len(coarse_d)} pts) = {np.mean(coarse_d):.4f}" if coarse_d else "  COARSE: n/a")
    print(f"  CASCADE scar DSC (mean over {len(casc_d)} pts) = {np.mean(casc_d):.4f}" if casc_d else "  CASCADE: n/a")
    if coarse_d and casc_d:
        print(f"  DELTA (cascade - coarse) = {np.mean(casc_d) - np.mean(coarse_d):+.4f}")
        print(f"  [ref: V2-XL OOF scar (all 40, pooled-mean) = 0.4112]")

if __name__ == "__main__":
    main()
