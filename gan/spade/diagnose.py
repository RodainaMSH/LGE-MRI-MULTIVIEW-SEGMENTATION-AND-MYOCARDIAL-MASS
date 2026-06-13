"""Diagnose deformed-synthesis quality: is the NEW-location scar genuinely muted, or is the gate's
contrast deflated by leftover brightness at the OLD scar location (now relabelled myo)?

For each deformed sample we measure, in tanh [-1,1] space:
  new_scar  = mean(fake at the deformed scar)
  myo_clean = mean(fake at myo pixels that are NOT within the old scar footprint)   <- true myo level
  old_leak  = mean(fake at the OLD scar footprint, which is now labelled myo)        <- style leakage
We compare contrast_clean = new_scar - myo_clean   vs   the gate's myo (includes old footprint).
Also dumps a PNG montage (real | recon | deformed-fake | deformed-label) for eyeballing.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan", required=True)
    ap.add_argument("--exclude_fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--png", default="gan/spade/runs/f0_diag.png")
    ap.add_argument("--netG", default="stylespade", choices=["stylespade", "spade"])
    args = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(11)
    pool = load_fold_slices(("SAX", "2CH", "4CH"), fold_exclude(args.exclude_fold), scar_only=True)
    opt = make_opt(is_train=False, gpu=True, netG=args.netG)
    model = SpadeModel(opt).to(dev)
    model.netG.load_state_dict(torch.load(args.gan, map_location=dev))
    model.netG.eval()

    c_gate, c_clean, leak, montage = [], [], [], []
    for _ in range(args.n):
        v, pid, z, a, g = pool[rng.integers(len(pool))]
        img128 = _resize(norm_img(a), 1)
        lab_o = np.rint(_resize(g.astype(np.float32), 0)).astype(np.int64)
        new = deform_scar(g, rng)
        if new is None:
            continue
        lab_n = np.rint(_resize(new.astype(np.float32), 0)).astype(np.int64)
        style = torch.from_numpy(img128[None, None]).float().to(dev)
        lt = torch.from_numpy(lab_n[None, None]).long().to(dev)
        with torch.no_grad():
            fk = model({"label": lt, "image": style}, "inference")[0, 0].cpu().numpy()
        new_scar = lab_n == 3
        old_scar = lab_o == 3
        myo_all = lab_n == 2
        myo_clean = myo_all & ~old_scar              # myo not contaminated by the old footprint
        if new_scar.sum() < 10 or myo_clean.sum() < 10:
            continue
        ns = fk[new_scar].mean()
        c_gate.append(ns - fk[myo_all].mean())
        c_clean.append(ns - fk[myo_clean].mean())
        if (old_scar & ~new_scar).sum() > 10:
            leak.append(fk[old_scar & ~new_scar].mean() - fk[myo_clean].mean())
        if len(montage) < 6:
            with torch.no_grad():
                rec = model({"label": torch.from_numpy(lab_o[None, None]).long().to(dev), "image": style}, "inference")[0, 0].cpu().numpy()
            montage.append((img128, rec, fk, lab_n))

    print(f"n={len(c_clean)}")
    print(f"  contrast (gate, myo incl. old footprint): {np.mean(c_gate):+.3f}")
    print(f"  contrast (CLEAN, myo excl. old footprint): {np.mean(c_clean):+.3f}   <- true new-scar contrast")
    print(f"  old-location leftover brightness vs clean myo: {np.mean(leak):+.3f}  ({100*np.mean(np.array(leak)>0.1):.0f}% still bright)")

    # montage PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(montage), 4, figsize=(10, 2.4 * len(montage)))
        cols = ["real (style)", "recon (orig label)", "deformed fake", "deformed label"]
        for i, (im, rec, fk, lb) in enumerate(montage):
            for j, M in enumerate([im, rec, fk]):
                ax[i, j].imshow(M, cmap="gray", vmin=-1, vmax=1)
            ax[i, 3].imshow(lb, cmap="nipy_spectral", vmin=0, vmax=4)
            for j in range(4):
                ax[i, j].axis("off")
                if i == 0:
                    ax[i, j].set_title(cols[j], fontsize=9)
        plt.tight_layout(); plt.savefig(args.png, dpi=90); print(f"saved {args.png}")
    except Exception as e:
        print("montage skipped:", e)


if __name__ == "__main__":
    main()
