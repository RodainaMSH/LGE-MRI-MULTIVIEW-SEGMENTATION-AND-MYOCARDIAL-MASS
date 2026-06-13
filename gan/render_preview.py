"""Render a GAN preview npz (real LGE | fake LGE | label) to a PNG grid for visual quality-gating.
Usage: python gan/render_preview.py gan/runs/sax_feas/preview_ep50.npz [out.png]
"""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else npz.replace(".npz", ".png")
d = np.load(npz)
real, fake, lab = d["real"], d["fake"], d["lab"]   # (N,1,H,W),(N,1,H,W),(N,H,W)
n = real.shape[0]
fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
if n == 1: ax = ax[None]
for i in range(n):
    ax[i, 0].imshow(real[i, 0], cmap="gray", vmin=-1, vmax=1); ax[i, 0].set_title("real LGE")
    ax[i, 1].imshow(fake[i, 0], cmap="gray", vmin=-1, vmax=1); ax[i, 1].set_title("FAKE LGE (GAN)")
    # label overlay: scar(3)=red, myo(2)=green, LVcav(1)=blue, RV(4)=yellow
    ov = np.zeros((*lab[i].shape, 3))
    ov[lab[i] == 1] = [0, 0, 1]; ov[lab[i] == 2] = [0, 1, 0]; ov[lab[i] == 3] = [1, 0, 0]; ov[lab[i] == 4] = [1, 1, 0]
    ax[i, 2].imshow(fake[i, 0], cmap="gray", vmin=-1, vmax=1); ax[i, 2].imshow(ov, alpha=0.4); ax[i, 2].set_title("fake + label (scar=red)")
    for j in range(3): ax[i, j].axis("off")
plt.tight_layout(); plt.savefig(out, dpi=70, bbox_inches="tight"); print("saved", out)
