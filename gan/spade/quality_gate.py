"""Synthesis quality gate for a trained per-fold StyleSPADE GAN.

Two checks, directly comparable to the v1 pix2pix gate (memory: v1 reconstruction +0.471, deformed +0.296):
  (A) RECONSTRUCTION — render the REAL (undeformed) label with its own slice as style; measure fake
      scar-vs-myo contrast and per-slice corr(real, fake). Tests raw fidelity.
  (B) DEFORMED — the one that matters for augmentation: deform the scar on-ring, render, measure fake
      scar-vs-myo contrast and % of samples where scar is brighter than myo. Must BEAT v1's +0.296.

GATE (printed as GATE=PASS/FAIL): deformed contrast > +0.30  AND  scar brighter than myo in >= 85%.

Usage: python gan/spade/quality_gate.py --gan gan/spade/runs/f0/G_final.pth --exclude_fold 0 --n 120
"""
import argparse
import numpy as np
import torch

import sys
sys.path.insert(0, "/home/youssef/projects/research/CMR-MULTI")
from gan.spade.make_opt import make_opt
from gan.spade.model import SpadeModel
from gan.spade.data import load_fold_slices, fold_exclude, norm_img, _resize
from gan.spade.deform import deform_scar


def contrast(fake, lab):
    """mean(fake at scar) - mean(fake at myo) on a 128x128 fake + its label."""
    scar, myo = (lab == 3), (lab == 2)
    if scar.sum() < 10 or myo.sum() < 10:
        return None
    return float(fake[scar].mean() - fake[myo].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan", required=True)
    ap.add_argument("--exclude_fold", type=int, default=-1)
    ap.add_argument("--views", default="SAX,2CH,4CH")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--netG", default="stylespade", choices=["stylespade", "spade"])
    ap.add_argument("--use_vae", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    rng = np.random.default_rng(7)
    excl = fold_exclude(args.exclude_fold)
    pool = load_fold_slices(tuple(args.views.split(",")), excl, scar_only=True)

    opt = make_opt(is_train=False, gpu=True, netG=args.netG, use_vae=args.use_vae)
    model = SpadeModel(opt).to(dev)
    model.netG.load_state_dict(torch.load(args.gan, map_location=dev))
    model.netG.eval()
    rmode = "sample" if args.use_vae else "inference"   # VAE renders from sampled z (diverse); recon-corr is then N/A

    rec_c, rec_corr, def_c, def_bright = [], [], [], []
    real_c = []
    for _ in range(args.n):
        v, pid, z, a, g = pool[rng.integers(len(pool))]
        img128 = _resize(norm_img(a), 1)
        lab128 = np.rint(_resize(g.astype(np.float32), 0)).astype(np.int64)
        style = torch.from_numpy(img128[None, None]).float().to(dev)
        # (A) reconstruction
        lt = torch.from_numpy(lab128[None, None]).long().to(dev)
        with torch.no_grad():
            fk = model({"label": lt, "image": style}, rmode)[0, 0].cpu().numpy()
        rc = contrast(fk, lab128); rcr = contrast(img128, lab128)
        if rc is not None:
            rec_c.append(rc); real_c.append(rcr)
            scar, myo = lab128 == 3, lab128 == 2
            both = np.concatenate([fk[scar], fk[myo]]); bothr = np.concatenate([img128[scar], img128[myo]])
            if both.std() > 1e-6 and bothr.std() > 1e-6:
                rec_corr.append(float(np.corrcoef(both, bothr)[0, 1]))
        # (B) deformed
        new = deform_scar(g, rng)
        if new is not None:
            nl = np.rint(_resize(new.astype(np.float32), 0)).astype(np.int64)
            lt2 = torch.from_numpy(nl[None, None]).long().to(dev)
            with torch.no_grad():
                fk2 = model({"label": lt2, "image": style}, rmode)[0, 0].cpu().numpy()
            dc = contrast(fk2, nl)
            if dc is not None:
                def_c.append(dc); def_bright.append(dc > 0)

    print(f"[gate fold {args.exclude_fold}]  n_rec={len(rec_c)} n_def={len(def_c)}")
    print(f"  (A) reconstruction: fake scar-myo contrast {np.mean(rec_c):+.3f}  (real {np.mean(real_c):+.3f})  corr {np.mean(rec_corr):.3f}")
    print(f"  (B) DEFORMED:       fake scar-myo contrast {np.mean(def_c):+.3f}   scar>myo in {100*np.mean(def_bright):.0f}%   (v1 was +0.296)")
    gate = (np.mean(def_c) > 0.30) and (np.mean(def_bright) >= 0.85)
    print(f"GATE={'PASS' if gate else 'FAIL'}  (need deformed contrast>+0.30 AND scar>myo>=85%)")


if __name__ == "__main__":
    main()
