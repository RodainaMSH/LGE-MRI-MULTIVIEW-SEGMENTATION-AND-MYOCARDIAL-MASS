"""Train a per-fold StyleSPADE GAN on our LGE (leakage-clean: only that fold's 32 train patients).

Mirrors their trainer (trainers/pix2pix_trainer.py): each iteration runs ONE generator step then ONE
discriminator step (D_steps_per_G=1), Adam with TTUR (G_lr=lr/2, D_lr=lr*2), then a linear LR decay over
the final `niter_decay` epochs (their update_learning_rate). Hinge GAN + feature-matching + VGG perceptual.

Usage:
  python gan/spade/train_spade.py --views SAX,2CH,4CH --exclude_fold 0 \
      --niter 150 --niter_decay 50 --bs 5 --out gan/spade/runs/f0
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, "/home/youssef/projects/research/CMR-MULTI")
from gan.spade.make_opt import make_opt
from gan.spade.model import SpadeModel
from gan.spade.data import SpadeSlices, fold_exclude


def update_lr(epoch, niter, niter_decay, lr, optG, optD, no_ttur=False):
    """Linear decay to 0 over the last niter_decay epochs (SPADE schedule)."""
    if epoch <= niter:
        new_lr = lr
    else:
        new_lr = lr * (niter + niter_decay - epoch) / max(niter_decay, 1)
    g_lr = new_lr if no_ttur else new_lr / 2
    d_lr = new_lr if no_ttur else new_lr * 2
    for pg in optG.param_groups:
        pg["lr"] = g_lr
    for pg in optD.param_groups:
        pg["lr"] = d_lr
    return new_lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="SAX,2CH,4CH")
    ap.add_argument("--exclude_fold", type=int, default=-1)
    ap.add_argument("--niter", type=int, default=150)
    ap.add_argument("--niter_decay", type=int, default=50)
    ap.add_argument("--bs", type=int, default=5)
    ap.add_argument("--netG", default="stylespade", choices=["stylespade", "spade"])
    ap.add_argument("--use_vae", action="store_true", help="SPADE-VAE (v3): encoder->latent + KLD; sample z at gen")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    views = tuple(args.views.split(","))
    excl = fold_exclude(args.exclude_fold)
    print(f"[SPADE fold {args.exclude_fold}] trains on {len(set(range(1,48))-excl)} patients (excluded {sorted(excl)})", flush=True)

    ds = SpadeSlices(views=views, exclude_pids=excl)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)

    opt = make_opt(is_train=True, gpu=True, netG=args.netG, use_vae=args.use_vae)
    model = SpadeModel(opt).to(dev)
    optG, optD = model.create_optimizers(opt)
    total = args.niter + args.niter_decay

    for ep in range(1, total + 1):
        lr = update_lr(ep, args.niter, args.niter_decay, opt.lr, optG, optD, opt.no_TTUR)
        agg = {}
        nb = 0
        for data in dl:
            # ---- generator step ----
            optG.zero_grad()
            g_losses, fake = model(data, "generator")
            g_loss = sum(v.mean() for v in g_losses.values())
            g_loss.backward()
            optG.step()
            # ---- discriminator step ----
            optD.zero_grad()
            d_losses = model(data, "discriminator")
            d_loss = sum(v.mean() for v in d_losses.values())
            d_loss.backward()
            optD.step()
            for k, v in {**g_losses, **d_losses}.items():
                agg[k] = agg.get(k, 0.0) + float(v.mean())
            nb += 1
        if ep % 10 == 0 or ep == 1:
            msg = "  ".join(f"{k} {agg[k]/nb:.3f}" for k in sorted(agg))
            print(f"epoch {ep}/{total}  lr {lr:.2e}  {msg}", flush=True)
            torch.save(model.netG.state_dict(), out / "G_latest.pth")
            # preview for quality eyeball (real | fake | label), no deformation here
            model.netG.eval()
            with torch.no_grad():
                d = next(iter(dl))
                fk = model({"label": d["label"], "image": d["image"]}, "inference")
            model.netG.train()
            np.savez(out / f"preview_ep{ep}.npz",
                     real=d["image"].numpy(), fake=fk.cpu().numpy(), lab=d["label"].numpy())
    torch.save(model.netG.state_dict(), out / "G_final.pth")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
