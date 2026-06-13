"""Stage 3: generate synthetic scar-augmented cases (ALL VIEWS) for seg-training augmentation.
For each synthetic case: take a real label of some view, transplant scar to a new wall location, GAN-synthesize
the LGE, resize image+label to that VIEW's real geometry, save as a single-slice Dataset case (image _0000 + label)
with the view's real affine. nnU-Net z-scores at train time, so the GAN's [-1,1] scale is fine.

Per-fold leakage-clean: --exclude_fold k loads templates ONLY from that fold's 32 train patients (excludes the 8
fold-test patients AND the held-out 7-val 41-47).  Counts split across views in proportion to real scar slices.

Usage: PYTHONPATH=. python gan/gen_synthetic.py --gan gan/runs/allv_fK/G_final.pth --n 300 \
         --views SAX,2CH,4CH --out nnunet/nnUNet_raw/Dataset021_LGEganclean --prefix SYNfK --exclude_fold K
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, nibabel as nib
sys.path.insert(0, "gan")
import torch
from train_pix2pix import UNetGen, _resize, NCLASS, SIZE, RAW
from augment_labels import scar_pool, transplant, lab_to_onehot

ROOT = Path("/home/youssef/projects/research/CMR-MULTI")


def view_geom(view):
    f = sorted((RAW / "imagesTr").glob(f"{view}_*_0000.nii.gz"))[0]
    ni = nib.load(str(f))
    return ni.affine, ni.header, np.asarray(ni.dataobj).shape[:2]


def load_view_slices(view, excl):
    """(img2d, lab2d) native-res slices for one view, excluding the held-out patients."""
    out = []
    for lab in sorted((RAW / "labelsTr").glob(f"{view}_*.nii.gz")):
        pid = int(lab.name.replace(".nii.gz", "").split("_")[1])
        if pid in excl:
            continue
        img = RAW / "imagesTr" / f"{view}_{pid:03d}_0000.nii.gz"
        if not img.exists():
            continue
        g = np.asarray(nib.load(str(lab)).dataobj).astype(np.int64)
        a = np.asarray(nib.load(str(img)).dataobj).astype(np.float32)
        for z in range(a.shape[2]):
            out.append((a[:, :, z], g[:, :, z]))
    return out


def gen_for_view(view, n_view, excl, G, dev, out, prefix, start):
    """Generate n_view synthetic cases for one view; returns count made."""
    slices = load_view_slices(view, excl)
    pool = scar_pool(slices)
    n_scar = sum(1 for _, g in slices if (g == 3).sum() > 0)
    aff, hdr, (H, W) = view_geom(view)
    if len(pool) == 0 or len(slices) == 0:
        print(f"[gen] {view}: NO templates/pool (excl {len(excl)}) -> skip", flush=True)
        return 0
    print(f"[gen] {view}: {len(slices)} templates ({n_scar} scar), pool {len(pool)}, geom {H}x{W} -> target {n_view}", flush=True)
    rng = np.random.default_rng(1234 + start + hash(view) % 1000)
    made = 0; tries = 0
    while made < n_view and tries < n_view * 25:
        tries += 1
        _, g = slices[rng.integers(len(slices))]
        newg = transplant(g, pool, rng)
        if newg is None:
            continue
        with torch.no_grad():
            fake = G(lab_to_onehot(newg)[None].to(dev))[0, 0].cpu().numpy()   # 256x256 in [-1,1]
        img = _resize(fake, max(H, W), 1)[:H, :W]
        lab = np.zeros((H, W), np.uint8)
        for c in range(1, NCLASS):
            lab[_resize((newg == c).astype(np.float32), max(H, W), 0)[:H, :W] > 0.5] = c
        if (lab == 3).sum() < 15:
            continue
        cid = f"{prefix}_{view}_{made:04d}"
        nib.save(nib.Nifti1Image(img[..., None].astype(np.float32), aff, hdr), str(out / "imagesTr" / f"{cid}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(lab[..., None], aff, hdr), str(out / "labelsTr" / f"{cid}.nii.gz"))
        made += 1
        if made % 50 == 0:
            print(f"    {view} {made}/{n_view}", flush=True)
    print(f"[gen] {view}: wrote {made} (tries {tries})", flush=True)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan", required=True)
    ap.add_argument("--n", type=int, default=300)                  # total across views
    ap.add_argument("--views", default="SAX,2CH,4CH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="SYNf0")
    ap.add_argument("--exclude_fold", type=int, default=-1)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    (out / "imagesTr").mkdir(parents=True, exist_ok=True); (out / "labelsTr").mkdir(parents=True, exist_ok=True)
    G = UNetGen().to(dev); G.load_state_dict(torch.load(args.gan, map_location=dev)); G.eval()
    views = tuple(args.views.split(","))
    # leakage-clean exclude set = all patients NOT in this fold's train (= fold-test + 41-47)
    excl = set()
    if args.exclude_fold >= 0:
        sp = json.load(open(ROOT / "nnunet/nnUNet_preprocessed/Dataset000_LGEgeneralist/splits_final.json"))
        train_pids = {int(c.split("_")[1]) for c in sp[args.exclude_fold]["train"]}
        all_pids = {int(p.name.replace(".nii.gz", "").split("_")[1]) for p in (RAW / "labelsTr").glob("SAX_*.nii.gz")}
        excl = all_pids - train_pids
    # split n across views in proportion to each view's real scar-slice availability
    scar_counts = {}
    for v in views:
        sl = load_view_slices(v, excl)
        scar_counts[v] = max(1, sum(1 for _, g in sl if (g == 3).sum() > 0))
    tot = sum(scar_counts.values())
    per = {v: int(round(args.n * scar_counts[v] / tot)) for v in views}
    print(f"[gen] fold {args.exclude_fold}: excl {sorted(excl)}\n[gen] view scar counts {scar_counts} -> per-view targets {per}", flush=True)
    total = 0
    for v in views:
        total += gen_for_view(v, per[v], excl, G, dev, out, args.prefix, args.start)
    print(f"[gen] DONE fold {args.exclude_fold}: wrote {total} synthetic cases (prefix {args.prefix}) to {out}", flush=True)


if __name__ == "__main__":
    main()
