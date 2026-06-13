"""Per-view scar MASS from masks + multi-view combine (rebuilt, generalist-only).

Reads segmentation masks (default: the generalist masks from nnunet_predict.py)
and the ground-truth annotations, computes each LV view's scar pseudo-mass in
grams (scar voxels x voxel-volume x myocardial density), then combines the three
per-view numbers into one final scar mass per patient via common.mass_combine.

Writes a JSON in the same shape the validator + compare scripts expect:
    patients[pid] = {scar_mass_per_expert:{SAX,2CH,4CH}, true_scar_mass, scar_mass_final}
    ras_patients[pid] = {pred_ra_mass, true_ra_mass}

The masks (DSC) come from the plain generalist; this script is the ONLY place the
multi-view "MoE" survives — as a mass combiner (no expert models). Validate the
combiner choice with unified_eval/validate_mass_combine.py.

Usage (after nnunet_predict.py):
    PYTHONPATH=. python nnunet/nnunet_mass.py --split val \
        --view-dir-fmt "gen_{view}" --native 4CH --out nnunet/results/logs/mass_val_gen.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from common.metrics import MYOCARDIAL_DENSITY_G_PER_CM3, patient_mass_grams
from common.mass_combine import native_weighted

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "LGE_MULTI"
SPLIT_SUFFIX = {"train": "TR", "val": "VAL"}
LV_VIEWS = ("SAX", "2CH", "4CH")
SCAR, RA = 3, 1


def voxel_vol_cm3(nii) -> float:
    z = nii.header.get_zooms()
    return float(np.prod(z[:3])) / 1000.0


def split_pids(view: str, split: str) -> list[int]:
    d = DATA / f"{view}_{SPLIT_SUFFIX[split]}" / "image"
    return sorted(int(p.name.split("_")[-1].split(".")[0])
                  for p in d.glob(f"LGE_{view}_*.nii.gz")) if d.exists() else []


def mass_of(mask_path: Path, anno_path: Path, cls: int):
    """Return (pred_mass_g, true_mass_g) for one view/patient, or None if missing."""
    if not mask_path.exists() or not anno_path.exists():
        return None
    pm = nib.load(str(mask_path)); gm = nib.load(str(anno_path))
    pred = np.asarray(pm.dataobj).astype(int); gt = np.asarray(gm.dataobj).astype(int)
    if pred.shape != gt.shape:
        print(f"  shape mismatch {mask_path.name}: {pred.shape} vs {gt.shape} -> skip")
        return None
    vv = voxel_vol_cm3(gm)  # pred & gt share geometry
    return (patient_mass_grams(pred, vv, [cls]), patient_mass_grams(gt, vv, [cls]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--pred-root", default=None, help="default: nnunet/predictions/<split>")
    ap.add_argument("--view-dir-fmt", default="gen_{view}",
                    help="subdir per view holding {VIEW}_{pid}.nii.gz (e.g. gen_{view} or expert_{view})")
    ap.add_argument("--native", default="4CH")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred_root = Path(args.pred_root) if args.pred_root else (ROOT / "nnunet" / "predictions" / args.split)
    suffix = SPLIT_SUFFIX[args.split]

    def mask_path(view, pid):
        return pred_root / args.view_dir_fmt.format(view=view) / f"{view}_{pid:03d}.nii.gz"

    def anno_path(view, pid):
        return DATA / f"{view}_{suffix}" / "anno" / f"LGE_{view}_{pid:03d}.nii.gz"

    patients = {}
    for pid in sorted(set().union(*[set(split_pids(v, args.split)) for v in LV_VIEWS])):
        per_view, true_scar = {}, None
        for view in LV_VIEWS:
            r = mass_of(mask_path(view, pid), anno_path(view, pid), SCAR)
            if r is None:
                continue
            per_view[view] = r[0]
            true_scar = r[1] if view == "SAX" else true_scar if true_scar is not None else r[1]
        if not per_view:
            continue
        # true scar from SAX (3D anchor) if present, else any available view's truth
        if true_scar is None:
            for view in LV_VIEWS:
                r = mass_of(mask_path(view, pid), anno_path(view, pid), SCAR)
                if r is not None:
                    true_scar = r[1]; break
        patients[str(pid)] = {
            "scar_mass_per_expert": per_view,
            "scar_mass_final": float(native_weighted(per_view, args.native)),
            "true_scar_mass": float(true_scar if true_scar is not None else 0.0),
        }

    ras = {}
    for pid in split_pids("RAS", args.split):
        r = mass_of(mask_path("RAS", pid), anno_path("RAS", pid), RA)
        if r is not None:
            ras[str(pid)] = {"pred_ra_mass": r[0], "true_ra_mass": r[1]}

    out = {
        "config": {"backbone": "nnU-Net v2 generalist (D000, 2d)", "split": args.split,
                   "combiner": f"native_weighted({args.native})", "scar_class": SCAR,
                   "density_g_per_cm3": MYOCARDIAL_DENSITY_G_PER_CM3,
                   "pred_root": str(pred_root), "view_dir_fmt": args.view_dir_fmt},
        "patients": patients, "ras_patients": ras,
    }
    out_path = Path(args.out) if args.out else (ROOT / "nnunet" / "results" / "logs" / f"mass_{args.split}_gen.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(patients)} LV + {len(ras)} RAS patients -> {out_path}")
    print(f"Validate the combiner: PYTHONPATH=. python unified_eval/validate_mass_combine.py "
          f"--mass-json {out_path} --native {args.native}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
