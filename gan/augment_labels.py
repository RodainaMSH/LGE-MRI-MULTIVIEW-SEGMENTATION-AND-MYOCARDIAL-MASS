"""Stage 2: scar-transplant label augmentation + GAN generation test.
Creates NEW label maps by moving a real scar shape to a NEW location within the myocardial wall, then has the
GAN synthesize the LGE. Tests the key question: does the GAN render scar in UNSEEN locations as realistic bright
tissue (the augmentation-value test), or does it only reconstruct training scar?

Builds a pool of real scar shapes; transplants onto target slices (scar ⊂ wall, mutually exclusive with myo).
Usage: python gan/augment_labels.py --gan gan/runs/sax_feas/G_final.pth --test
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage
sys.path.insert(0, "gan")
import torch
from train_pix2pix import SAXSlices, UNetGen, _resize, NCLASS, SIZE, RAW

def load_slices():
    """Return list of (img2d, lab2d) at native res for SAX training slices."""
    out = []
    for lab in sorted((RAW / "labelsTr").glob("SAX_*.nii.gz")):
        pid = int(lab.name.replace(".nii.gz", "").split("_")[1])
        img = RAW / "imagesTr" / f"SAX_{pid:03d}_0000.nii.gz"
        if not img.exists(): continue
        g = np.asarray(nib.load(str(lab)).dataobj).astype(np.int64)
        a = np.asarray(nib.load(str(img)).dataobj).astype(np.float32)
        for z in range(a.shape[2]):
            out.append((a[:, :, z], g[:, :, z]))
    return out

def scar_pool(slices, min_px=40):
    """Binary scar-shape masks (bbox-cropped) from real slices."""
    pool = []
    for _, g in slices:
        scar = (g == 3)
        lbl, n = ndimage.label(scar)
        for k in range(1, n + 1):
            m = lbl == k
            if m.sum() < min_px: continue
            ys, xs = np.where(m)
            pool.append(m[ys.min():ys.max()+1, xs.min():xs.max()+1].copy())
    return pool

def transplant(lab, pool, rng):
    """Move scar to a NEW random location within the wall (myo∪scar). Returns new label (same shape)."""
    wall = (lab == 2) | (lab == 3)
    if wall.sum() < 50: return None
    new = np.where(lab == 3, 2, lab)             # erase existing scar -> myo
    shape = pool[rng.integers(len(pool))]
    wy, wx = np.where(wall)
    cy, cx = wy[rng.integers(len(wy))], wx[rng.integers(len(wx))]   # random wall anchor
    sh, sw = shape.shape
    y0, x0 = cy - sh // 2, cx - sw // 2
    placed = np.zeros_like(wall)
    ys0, ys1 = max(y0, 0), min(y0 + sh, lab.shape[0]); xs0, xs1 = max(x0, 0), min(x0 + sw, lab.shape[1])
    if ys1 <= ys0 or xs1 <= xs0: return None
    placed[ys0:ys1, xs0:xs1] = shape[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    newscar = placed & wall                      # scar only where it lands ON the wall
    if newscar.sum() < 20: return None
    new[newscar] = 3
    return new

def lab_to_onehot(g):
    return torch.from_numpy(np.stack([_resize((g == c).astype(np.float32), SIZE, 0) for c in range(NCLASS)])).float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan", default="gan/runs/sax_feas/G_final.pth")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = UNetGen().to(dev); G.load_state_dict(torch.load(args.gan, map_location=dev)); G.eval()
    slices = load_slices(); pool = scar_pool(slices)
    print(f"[Stage2] {len(slices)} slices, scar-shape pool = {len(pool)} shapes")
    rng = np.random.default_rng(0)
    # generate augmented examples + check the TRANSPLANTED scar renders bright
    fcon = []; previews = []
    tries = 0
    while len(fcon) < 120 and tries < 2000:
        tries += 1
        _, g = slices[rng.integers(len(slices))]
        newg = transplant(g, pool, rng)
        if newg is None: continue
        with torch.no_grad():
            fake = G(lab_to_onehot(newg)[None].to(dev))[0, 0].cpu().numpy()
        fr = _resize((newg == 3).astype(np.float32), SIZE, 0) > 0.5
        mr = _resize((newg == 2).astype(np.float32), SIZE, 0) > 0.5
        if fr.sum() < 20 or mr.sum() < 20: continue
        fcon.append(fake[fr].mean() - fake[mr].mean())
        if len(previews) < 4: previews.append((newg.copy(), fake.copy(), fr, mr))
    fcon = np.array(fcon)
    print(f"[Stage2] TRANSPLANTED (new-location) scar−myo contrast over {len(fcon)} synth slices:")
    print(f"  mean {fcon.mean():+.3f}  median {np.median(fcon):+.3f}  (scar>myo in {100*(fcon>0).mean():.0f}%)")
    print(f"  (real scar−myo contrast was +0.477 — GAN GENERALIZES to new scar if this is similarly positive)")
    if args.test:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(previews), 2, figsize=(10, 5 * len(previews)))
        for r, (newg, fake, fr, mr) in enumerate(previews):
            ov = np.zeros((SIZE, SIZE, 3)); ov[_resize((newg==2).astype(np.float32),SIZE,0)>0.5]=[0,1,0]; ov[fr]=[1,0,0]
            ax[r,0].imshow(fake,cmap="gray",vmin=-1,vmax=1); ax[r,0].set_title("FAKE LGE (transplanted scar)"); ax[r,0].axis("off")
            ax[r,1].imshow(fake,cmap="gray",vmin=-1,vmax=1); ax[r,1].imshow(ov,alpha=0.35); ax[r,1].set_title("fake + NEW scar(red)/myo(green)"); ax[r,1].axis("off")
        plt.tight_layout(); plt.savefig("/tmp/gan_transplant.png", dpi=100, bbox_inches="tight"); print("saved /tmp/gan_transplant.png")

if __name__ == "__main__":
    main()
