"""Per-view quality gate for an all-views per-fold GAN. For each view, transplant scar to NEW wall locations,
render with the GAN, and check the synthetic scar comes out BRIGHTER than surrounding myocardium (the property
the seg model needs). A view PASSES if scar>myo in >=85% of synthetic slices AND mean contrast >= +0.10.
Views that fail should be dropped from augmentation (fall back to SAX-only).

Usage: PYTHONPATH=. python gan/quality_gate.py --gan gan/runs/allv_f0/G_final.pth --views SAX,2CH,4CH --exclude_fold 0
Prints a PASS/FAIL line per view and a final 'GATE_VIEWS=SAX,2CH' summary listing the views that passed.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "gan")
import torch
from train_pix2pix import UNetGen, _resize, NCLASS, SIZE, RAW
from augment_labels import scar_pool, transplant, lab_to_onehot
from gen_synthetic import load_view_slices

ROOT = Path("/home/youssef/projects/research/CMR-MULTI")
PASS_FRAC, PASS_CONTRAST = 0.85, 0.10


def gate_view(view, excl, G, dev, n=100):
    slices = load_view_slices(view, excl)
    pool = scar_pool(slices)
    if len(pool) == 0 or len(slices) == 0:
        print(f"  {view}: NO data -> FAIL"); return False
    rng = np.random.default_rng(7)
    con = []; tries = 0
    while len(con) < n and tries < n * 25:
        tries += 1
        _, g = slices[rng.integers(len(slices))]
        newg = transplant(g, pool, rng)
        if newg is None:
            continue
        with torch.no_grad():
            fake = G(lab_to_onehot(newg)[None].to(dev))[0, 0].cpu().numpy()
        fr = _resize((newg == 3).astype(np.float32), SIZE, 0) > 0.5
        mr = _resize((newg == 2).astype(np.float32), SIZE, 0) > 0.5
        if fr.sum() < 20 or mr.sum() < 20:
            continue
        con.append(fake[fr].mean() - fake[mr].mean())
    con = np.array(con)
    frac = float((con > 0).mean()); mean = float(con.mean())
    ok = frac >= PASS_FRAC and mean >= PASS_CONTRAST
    print(f"  {view}: scar>myo {100*frac:.0f}% | mean contrast {mean:+.3f} | n={len(con)} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan", required=True)
    ap.add_argument("--views", default="SAX,2CH,4CH")
    ap.add_argument("--exclude_fold", type=int, default=-1)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = UNetGen().to(dev); G.load_state_dict(torch.load(args.gan, map_location=dev)); G.eval()
    excl = set()
    if args.exclude_fold >= 0:
        sp = json.load(open(ROOT / "nnunet/nnUNet_preprocessed/Dataset000_LGEgeneralist/splits_final.json"))
        train_pids = {int(c.split("_")[1]) for c in sp[args.exclude_fold]["train"]}
        all_pids = {int(p.name.replace(".nii.gz", "").split("_")[1]) for p in (RAW / "labelsTr").glob("SAX_*.nii.gz")}
        excl = all_pids - train_pids
    print(f"[quality gate] GAN {args.gan} (fold {args.exclude_fold}):")
    passed = [v for v in args.views.split(",") if gate_view(v, excl, G, dev)]
    print(f"GATE_VIEWS={','.join(passed)}")


if __name__ == "__main__":
    main()
