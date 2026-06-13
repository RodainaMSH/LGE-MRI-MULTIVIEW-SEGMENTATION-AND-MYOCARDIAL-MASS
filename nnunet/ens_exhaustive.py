"""Exhaustive multi-arch ensemble search over CACHED per-fold softmax probs (/tmp/ens/probs).
Pool = ALL tested 5-fold-complete models. For EVERY non-empty subset, soft-average (equal weight),
argmax, score on the OFFICIAL Task2 DSC (mean Dice per structure/view/patient) AND the mass clinical
term (0.5*max(0,vPCC)+0.5/(1+RAE_cal), SAX scar grams). Mask & mass are decoupled (separate submission).
Reports the BEST combo of EACH SIZE (2,3,...,N) + overall top-15, and writes top combos to JSON for
the 7-val validation phase. Geometry verified against the saved .nii.gz. Equal-weight only.
"""
import os, glob, json, itertools, sys
import numpy as np, nibabel as nib, pandas as pd

ROOT = "/home/youssef/projects/research/CMR-MULTI"
SP = "/tmp/ens/probs"
DATA = f"{ROOT}/LGE_MULTI"
DENS, SCAR = 1.05, 3
POOL = {
    "XL": "nnUNetTrainerSMP_effv2xl", "DN": "nnUNetTrainerSMP_densenet264",
    "L": "nnUNetTrainerSMP_effv2l", "B7": "nnUNetTrainerSMP_effb7",
    "ST": "nnUNetTrainerScarTversky", "PC": "nnUNetTrainer_100epochs",
    "RIN": "nnUNetTrainerRIN_densenet121", "HM": "nnUNetTrainerHieraMedSAM2",
    "SAM": "nnUNetTrainerSAM2large", "GA": "ganaug_dn264", "SP": "spade_dn264",
}
# only keep models that actually have cached probs
KEYS = [k for k in POOL if os.path.isdir(f"{SP}/{POOL[k]}/fold_0") and glob.glob(f"{SP}/{POOL[k]}/fold_*/*.npz")]
VIEW_FG = {"SAX": [1, 2, 3, 4], "2CH": [1, 2, 3], "4CH": [1, 2, 3, 4]}

case_fold = {}
for k in range(5):
    for p in glob.glob(f"{SP}/{POOL['XL']}/fold_{k}/*.npz"):
        case_fold[os.path.basename(p)[:-4]] = k
cases = sorted(case_fold)
print(f"{len(cases)} OOF cases; pool ({len(KEYS)}) = {KEYS}", flush=True)

def anno(view, pid):
    return f"{DATA}/{view}_TR/anno/LGE_{view}_{pid:03d}.nii.gz"

df = pd.read_excel(f"{DATA}/dataset_lge.xlsx", sheet_name="SAX")
true_mass = dict(zip(df["编号"].astype(int), df["Scar_Quality"].astype(float)))

# geometry transpose (verify against saved nii.gz)
c0 = cases[0]; k0 = case_fold[c0]
pr0 = np.load(f"{SP}/{POOL['XL']}/fold_{k0}/{c0}.npz")["probabilities"]
nii0 = np.asarray(nib.load(f"{SP}/{POOL['XL']}/fold_{k0}/{c0}.nii.gz").dataobj)
am0 = pr0.argmax(0)
Tf = next((perm for perm in itertools.permutations(range(3))
           if am0.transpose(perm).shape == nii0.shape and np.array_equal(am0.transpose(perm), nii0.astype(am0.dtype))), None)
assert Tf is not None
print(f"transpose={Tf}; building subsets...", flush=True)

subsets = [s for r in range(1, len(KEYS) + 1) for s in itertools.combinations(KEYS, r)]
print(f"{len(subsets)} subsets to score", flush=True)
dsc_sum = {s: 0.0 for s in subsets}; dsc_n = {s: 0 for s in subsets}
predmass = {s: {} for s in subsets}

for ci, case in enumerate(cases):
    k = case_fold[case]
    view, pid = case.split("_")[0], int(case.split("_")[1])
    gt = np.asarray(nib.load(anno(view, pid)).dataobj).astype(np.int16)
    vv = float(np.prod(nib.load(anno(view, pid)).header.get_zooms()[:3])) / 1000.0
    P = {key: np.load(f"{SP}/{POOL[key]}/fold_{k}/{case}.npz")["probabilities"].astype(np.float32) for key in KEYS}
    fg = VIEW_FG[view]
    for s in subsets:
        ssum = P[s[0]].copy()
        for key in s[1:]:
            ssum += P[key]
        pred = ssum.argmax(0).transpose(Tf).astype(np.int16)
        if pred.shape != gt.shape:
            continue
        for c in fg:
            p, g = (pred == c), (gt == c)
            d = p.sum() + g.sum()
            dsc_sum[s] += 1.0 if d == 0 else 2 * (p & g).sum() / d
            dsc_n[s] += 1
        if view == "SAX":
            predmass[s][pid] = float((pred == SCAR).sum() * vv * DENS)
    if (ci + 1) % 20 == 0:
        print(f"  ...{ci+1}/{len(cases)}", flush=True)

def vpcc(p, t):
    return float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 and p.std() and t.std() else float("nan")
def best_rae(pred, true):
    nz = true > 1e-6
    if not nz.any(): return float("nan")
    return min(float(np.mean(np.abs(np.clip(a * pred[nz], 0, None) - true[nz]) / true[nz])) for a in np.arange(0.05, 3.001, 0.01))

rows = []
for s in subsets:
    if dsc_n[s] == 0: continue
    d = dsc_sum[s] / dsc_n[s]
    pids = sorted(predmass[s])
    pred = np.array([predmass[s][p] for p in pids]); tru = np.array([true_mass[p] for p in pids])
    v = vpcc(pred, tru); r = best_rae(pred, tru); clin = 0.5 * max(0, v) + 0.5 / (1 + r)
    rows.append({"m": list(s), "size": len(s), "dsc": d, "vpcc": v, "rae": r, "massclin": clin})

print("\n========= BEST DSC (mask) BY COMBO SIZE =========")
print(f"{'size':>4}  {'best combo':<34}{'DSC':>8}{'massClin':>9}")
for sz in range(1, len(KEYS) + 1):
    cand = [x for x in rows if x["size"] == sz]
    if not cand: continue
    b = max(cand, key=lambda x: x["dsc"])
    print(f"{sz:>4}  {'+'.join(b['m']):<34}{b['dsc']:>8.4f}{b['massclin']:>9.3f}")

print("\n========= BEST MASS (clinical) BY COMBO SIZE =========")
print(f"{'size':>4}  {'best combo':<34}{'massClin':>9}{'vPCC':>7}{'DSC':>8}")
for sz in range(1, len(KEYS) + 1):
    cand = [x for x in rows if x["size"] == sz]
    if not cand: continue
    b = max(cand, key=lambda x: x["massclin"])
    print(f"{sz:>4}  {'+'.join(b['m']):<34}{b['massclin']:>9.3f}{b['vpcc']:>7.3f}{b['dsc']:>8.4f}")

print("\n========= OVERALL TOP 15 by DSC =========")
for x in sorted(rows, key=lambda x: -x["dsc"])[:15]:
    print(f"  {'+'.join(x['m']):<36}DSC {x['dsc']:.4f}  massClin {x['massclin']:.3f} (vPCC {x['vpcc']:.3f}, RAE {x['rae']:.3f})")

bestD = max(rows, key=lambda x: x["dsc"]); bestM = max(rows, key=lambda x: x["massclin"])
print(f"\nBEST DECOUPLED: mask={'+'.join(bestD['m'])} (DSC {bestD['dsc']:.4f}) + mass={'+'.join(bestM['m'])} (clin {bestM['massclin']:.3f})")
print(f"  Task2 = 0.7*{bestD['dsc']:.4f} + 0.3*{bestM['massclin']:.3f} = {0.7*bestD['dsc']+0.3*bestM['massclin']:.4f}")

# save top combos for the 7-val validation phase
top = {
    "pool": KEYS, "trainer": POOL,
    "top_dsc": [x for x in sorted(rows, key=lambda x: -x["dsc"])[:8]],
    "top_mass": [x for x in sorted(rows, key=lambda x: -x["massclin"])[:8]],
}
json.dump(top, open(f"{ROOT}/nnunet/ens_top_combos.json", "w"), indent=2)
print(f"\nwrote top combos -> nnunet/ens_top_combos.json")
