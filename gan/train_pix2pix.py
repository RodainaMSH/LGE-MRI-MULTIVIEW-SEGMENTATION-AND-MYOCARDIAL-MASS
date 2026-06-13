"""Label->LGE conditional GAN (pix2pix) for scar-augmentation (2026-06-10, Youssef's "GAN lever").
Learns to synthesize a realistic LGE slice from a 5-class segmentation label map (bg/LVcav/LVmyo/scar/RVcav).
Then (later stages) we generate synthetic LGE for scar-augmented label maps to expand the N=40 train set.

STAGE 1 = train this GAN + VALIDATE synthesis quality before doing any seg augmentation. If 470 SAX slices
can't make realistic LGE, we stop here.

Design: standard pix2pix — U-Net generator (5ch label -> 1ch LGE, tanh), 70x70 PatchGAN discriminator
(6ch [label|image] -> patch), LSGAN adv loss + L1(λ=100), Adam 2e-4 (0.5,0.999). Per-image percentile
normalization to [-1,1] (nnU-Net z-scores at seg-train time, so absolute scale is irrelevant).

Usage: python gan/train_pix2pix.py --views SAX --epochs 200 --out gan/runs/sax --exclude_fold -1
  --exclude_fold k  -> hold out fold k's patients (leakage-clean per-fold GAN); -1 = use all (feasibility).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, nibabel as nib
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/home/youssef/projects/research/CMR-MULTI")
RAW = ROOT / "nnunet/nnUNet_raw/Dataset000_LGEgeneralist"
NCLASS = 5
SIZE = 256

def _resize(a, size, order):
    import torch.nn.functional as F
    t = torch.from_numpy(a.astype(np.float32))[None, None]
    mode = "bilinear" if order == 1 else "nearest"
    kw = dict(align_corners=False) if order == 1 else {}
    return F.interpolate(t, size=(size, size), mode=mode, **kw)[0, 0].numpy()

class SAXSlices(Dataset):
    """All 2D (label one-hot 5ch, LGE 1ch in [-1,1]) slices from the chosen views, optionally excluding a fold."""
    def __init__(self, views=("SAX",), exclude_pids=()):
        self.items = []
        for lab in sorted((RAW / "labelsTr").glob("*.nii.gz")):
            v, pid = lab.name.replace(".nii.gz", "").split("_")[0], int(lab.name.replace(".nii.gz", "").split("_")[1])
            if v not in views or pid in exclude_pids:
                continue
            img = RAW / "imagesTr" / f"{v}_{pid:03d}_0000.nii.gz"
            if not img.exists():
                continue
            g = np.asarray(nib.load(str(lab)).dataobj).astype(np.int64)
            a = np.asarray(nib.load(str(img)).dataobj).astype(np.float32)
            for z in range(a.shape[2]):
                self.items.append((a[:, :, z].copy(), g[:, :, z].copy()))
        print(f"[GAN data] {len(self.items)} slices from views={views} (excluded {len(exclude_pids)} pids)")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        a, g = self.items[i]
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)            # per-image robust scale -> [-1,1]
        a = np.clip(a, lo, hi); a = 2 * (a - lo) / max(hi - lo, 1e-6) - 1
        img = _resize(a, SIZE, 1)[None]                              # 1 x H x W
        oh = np.stack([_resize((g == c).astype(np.float32), SIZE, 0) for c in range(NCLASS)])  # 5 x H x W
        return torch.from_numpy(oh).float(), torch.from_numpy(img).float()

def _down(ci, co, norm=True):
    layers = [nn.Conv2d(ci, co, 4, 2, 1, bias=not norm)]
    if norm: layers.append(nn.BatchNorm2d(co))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)

def _up(ci, co, drop=False):
    layers = [nn.ConvTranspose2d(ci, co, 4, 2, 1, bias=False), nn.BatchNorm2d(co)]
    if drop: layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)

class UNetGen(nn.Module):
    """pix2pix U-Net: 5ch label -> 1ch LGE (tanh). 256->...->1->...->256 with skips."""
    def __init__(self, cin=NCLASS, cout=1, f=64):
        super().__init__()
        self.d1 = _down(cin, f, norm=False)   # 128
        self.d2 = _down(f, f * 2)             # 64
        self.d3 = _down(f * 2, f * 4)         # 32
        self.d4 = _down(f * 4, f * 8)         # 16
        self.d5 = _down(f * 8, f * 8)         # 8
        self.d6 = _down(f * 8, f * 8)         # 4
        self.d7 = _down(f * 8, f * 8)         # 2
        self.d8 = _down(f * 8, f * 8, norm=False)  # 1
        self.u1 = _up(f * 8, f * 8, drop=True)
        self.u2 = _up(f * 16, f * 8, drop=True)
        self.u3 = _up(f * 16, f * 8, drop=True)
        self.u4 = _up(f * 16, f * 8)
        self.u5 = _up(f * 16, f * 4)
        self.u6 = _up(f * 8, f * 2)
        self.u7 = _up(f * 4, f)
        self.u8 = nn.Sequential(nn.ConvTranspose2d(f * 2, cout, 4, 2, 1), nn.Tanh())
    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1)); u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1)); u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1)); u7 = self.u7(torch.cat([u6, d2], 1))
        return self.u8(torch.cat([u7, d1], 1))

class PatchD(nn.Module):
    """70x70 PatchGAN on [label(5)|image(1)] = 6ch."""
    def __init__(self, cin=NCLASS + 1, f=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, f, 4, 2, 1), nn.LeakyReLU(0.2, True),
            _down(f, f * 2), _down(f * 2, f * 4),
            nn.Conv2d(f * 4, f * 8, 4, 1, 1), nn.BatchNorm2d(f * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(f * 8, 1, 4, 1, 1))
    def forward(self, lab, img): return self.net(torch.cat([lab, img], 1))

def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    excl = set()
    if args.exclude_fold >= 0:
        splits = json.load(open(ROOT / "nnunet/nnUNet_preprocessed/Dataset000_LGEgeneralist/splits_final.json"))
        train_pids = {int(c.split("_")[1]) for c in splits[args.exclude_fold]["train"]}
        all_pids = {int(p.name.replace(".nii.gz", "").split("_")[1]) for p in (RAW / "labelsTr").glob("SAX_*.nii.gz")}
        excl = all_pids - train_pids   # exclude this fold's TEST patients AND the held-out 7-val (41-47) — train-only
    ds = SAXSlices(views=tuple(args.views.split(",")), exclude_pids=excl)
    print(f"[GAN] fold {args.exclude_fold}: trains on {len(set(range(1,48))-excl)} patients (excluded {sorted(excl)})")
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    G, D = UNetGen().to(dev), PatchD().to(dev)
    optG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    for ep in range(args.epochs):
        gl = dl_ = 0.0
        for lab, real in dl:
            lab, real = lab.to(dev), real.to(dev)
            fake = G(lab)
            # D
            optD.zero_grad()
            pr, pf = D(lab, real), D(lab, fake.detach())
            lossD = 0.5 * (mse(pr, torch.ones_like(pr)) + mse(pf, torch.zeros_like(pf)))
            lossD.backward(); optD.step()
            # G
            optG.zero_grad()
            pf = D(lab, fake)
            lossG = mse(pf, torch.ones_like(pf)) + 100.0 * l1(fake, real)
            lossG.backward(); optG.step()
            gl += lossG.item(); dl_ += lossD.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep+1}/{args.epochs}  G {gl/len(dl):.3f}  D {dl_/len(dl):.3f}", flush=True)
            torch.save(G.state_dict(), out / "G_latest.pth")
            # save a preview grid every 10 ep for quality eyeballing
            with torch.no_grad():
                lab, real = next(iter(dl)); lab, real = lab[:4].to(dev), real[:4].to(dev)
                fake = G(lab)
                np.savez(out / f"preview_ep{ep+1}.npz",
                         real=real.cpu().numpy(), fake=fake.cpu().numpy(), lab=lab.argmax(1).cpu().numpy())
    torch.save(G.state_dict(), out / "G_final.pth")
    print("DONE", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="SAX")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="gan/runs/sax")
    ap.add_argument("--exclude_fold", type=int, default=-1)
    train(ap.parse_args())
