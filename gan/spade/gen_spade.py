"""Generate synthetic (LGE image, label) cases for one fold using its trained StyleSPADE GAN.

For each synthetic sample: take a REAL scar-bearing slice of one of the fold's 32 train patients,
DEFORM its scar on-ring (gan/spade/deform.py — rotate*60 about LV centroid + elastic + dilation/opening),
then render with the GAN using that patient's OWN slice as the style image and the DEFORMED label as
anatomy:  fake = G( one_hot(deformed_label), style=real_slice ).
The style code is global (appearance), SPADE places the scar at the new on-ring location -> the scar is
rendered with the real slice's true brightness at a new, anatomically-plausible site (fixes v1's muted contrast).

Leakage: same train-only exclusion as training. Synthetic for fold k is tagged SYNfk_* and (downstream)
routed ONLY into fold k's training split.

Usage:
  python gan/spade/gen_spade.py --gan gan/spade/runs/f0/G_final.pth --exclude_fold 0 \
      --views SAX,2CH,4CH --n 300 --prefix SYNf0 --out_img <dir> --out_lab <dir>
"""
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch

import sys
sys.path.insert(0, "/home/youssef/projects/research/CMR-MULTI")
from gan.spade.make_opt import make_opt
from gan.spade.model import SpadeModel
from gan.spade.data import load_fold_slices, fold_exclude, norm_img, _resize, RAW, SIZE
from gan.spade.deform import deform_scar


def _resize_to(a, H, W, order):
    import torch.nn.functional as F
    t = torch.from_numpy(a.astype(np.float32))[None, None]
    if order == 1:
        return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    return F.interpolate(t, size=(H, W), mode="nearest")[0, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan", required=True)
    ap.add_argument("--exclude_fold", type=int, default=-1)
    ap.add_argument("--views", default="SAX,2CH,4CH")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out_img", required=True)
    ap.add_argument("--out_lab", required=True)
    ap.add_argument("--netG", default="stylespade", choices=["stylespade", "spade"])
    ap.add_argument("--use_vae", action="store_true", help="SPADE-VAE (v3): sample z~N(0,1) for appearance diversity")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    out_img = Path(args.out_img); out_img.mkdir(parents=True, exist_ok=True)
    out_lab = Path(args.out_lab); out_lab.mkdir(parents=True, exist_ok=True)
    views = tuple(args.views.split(","))
    rng = np.random.default_rng(args.seed + 1000 * (args.exclude_fold + 1))

    excl = fold_exclude(args.exclude_fold)
    pool = load_fold_slices(views, excl, scar_only=True)
    print(f"[gen fold {args.exclude_fold}] {len(pool)} scar slices available (excluded {sorted(excl)})", flush=True)
    assert pool, "no scar slices for this fold"

    opt = make_opt(is_train=False, gpu=True, netG=args.netG, use_vae=args.use_vae)
    model = SpadeModel(opt).to(dev)
    model.netG.load_state_dict(torch.load(args.gan, map_location=dev))
    model.netG.eval()
    render_mode = "sample" if args.use_vae else "inference"   # VAE: z~N(0,1) diversity; else style/label render

    made = 0
    tries = 0
    while made < args.n and tries < args.n * 20:
        tries += 1
        v, pid, z, a, g = pool[rng.integers(len(pool))]
        new_lab = deform_scar(g, rng)
        if new_lab is None:
            continue
        H, W = a.shape
        style = torch.from_numpy(_resize(norm_img(a), 1)[None, None]).float().to(dev)   # 1x1x128x128
        lab128 = torch.from_numpy(_resize(new_lab.astype(np.float32), 0)[None, None]).long().to(dev)
        with torch.no_grad():
            fake = model({"label": lab128, "image": style}, render_mode)[0, 0].cpu().numpy()  # 128x128 [-1,1]
        # back to the real slice geometry
        fake_full = _resize_to(fake, H, W, 1).astype(np.float32)
        lab_full = np.rint(_resize_to(new_lab.astype(np.float32), H, W, 0)).astype(np.uint8)
        if (lab_full == 3).sum() < 20:
            continue
        aff = nib.load(str(RAW / "imagesTr" / f"{v}_{pid:03d}_0000.nii.gz")).affine
        name = f"{args.prefix}_{v}_{made:04d}"
        nib.save(nib.Nifti1Image(fake_full[:, :, None], aff), str(out_img / f"{name}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(lab_full[:, :, None], aff), str(out_lab / f"{name}.nii.gz"))
        made += 1
    print(f"DONE generated {made} synthetic cases (prefix {args.prefix}, tries {tries})", flush=True)


if __name__ == "__main__":
    main()
