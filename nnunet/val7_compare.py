"""Phase C: validate candidate ensemble combos on the HELD-OUT 7-val (patients 41-47), side-by-side
with their OOF (40-patient) numbers. OOF probs = /tmp/ens/probs/<tr>/fold_k/<case>.npz (per-fold OOF);
7-val probs = /tmp/ens/val7/<tr>/<case>.npz (5-fold ensemble on 41-47). GT: TR anno + dataset_lge.xlsx
for OOF; VAL anno + dataset_lge_valid.xlsx for 7-val. Scores official DSC (mean per structure/view/patient)
+ mass clinical term. NOTE: 7-val n=7 (≈5 nonzero-scar) is NOISY — a confirmation check, not a selector.
"""
import os, glob, json, itertools
import numpy as np, nibabel as nib, pandas as pd

ROOT = "/home/youssef/projects/research/CMR-MULTI"; DATA = f"{ROOT}/LGE_MULTI"
DENS, SCAR = 1.05, 3
TR = {"XL": "nnUNetTrainerSMP_effv2xl", "DN": "nnUNetTrainerSMP_densenet264", "L": "nnUNetTrainerSMP_effv2l",
      "B7": "nnUNetTrainerSMP_effb7", "ST": "nnUNetTrainerScarTversky", "PC": "nnUNetTrainer_100epochs",
      "RIN": "nnUNetTrainerRIN_densenet121", "HM": "nnUNetTrainerHieraMedSAM2", "SAM": "nnUNetTrainerSAM2large",
      "GA": "ganaug_dn264", "SP": "spade_dn264"}
VIEW_FG = {"SAX": [1, 2, 3, 4], "2CH": [1, 2, 3], "4CH": [1, 2, 3, 4]}

# geometry transpose
import nibabel
c0 = "SAX_001"
pr0 = np.load(f"/tmp/ens/probs/{TR['XL']}/fold_0/{c0}.npz")["probabilities"]
nii0 = np.asarray(nib.load(f"/tmp/ens/probs/{TR['XL']}/fold_0/{c0}.nii.gz").dataobj)
Tf = next(p for p in itertools.permutations(range(3)) if pr0.argmax(0).transpose(p).shape == nii0.shape and np.array_equal(pr0.argmax(0).transpose(p), nii0.astype(np.int16)))

def cases_for(regime):
    if regime == "OOF":
        cf = {}
        for k in range(5):
            for p in glob.glob(f"/tmp/ens/probs/{TR['XL']}/fold_{k}/*.npz"):
                cf[os.path.basename(p)[:-4]] = k
        return cf
    return {os.path.basename(p)[:-4]: None for p in glob.glob(f"/tmp/ens/val7/{TR['XL']}/*.npz")}

def gt_anno(regime, view, pid):
    sub = "TR" if regime == "OOF" else "VAL"
    return f"{DATA}/{view}_{sub}/anno/LGE_{view}_{pid:03d}.nii.gz"

def true_mass(regime):
    f = "dataset_lge.xlsx" if regime == "OOF" else "dataset_lge_valid.xlsx"
    df = pd.read_excel(f"{DATA}/{f}", sheet_name="SAX")
    return dict(zip(df["编号"].astype(int), df["Scar_Quality"].astype(float)))

def probs_path(regime, tr, case, fold):
    return f"/tmp/ens/probs/{tr}/fold_{fold}/{case}.npz" if regime == "OOF" else f"/tmp/ens/val7/{tr}/{case}.npz"

def vpcc(p, t): return float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 and p.std() and t.std() else float("nan")
def best_rae(pred, true):
    nz = true > 1e-6
    if not nz.any(): return float("nan")
    return min(float(np.mean(np.abs(np.clip(a*pred[nz], 0, None)-true[nz])/true[nz])) for a in np.arange(0.05, 3.001, 0.01))

def eval_combo(members, regime, cf, tm):
    dsc, pm = [], {}
    for case, fold in cf.items():
        view, pid = case.split("_")[0], int(case.split("_")[1])
        gp = gt_anno(regime, view, pid)
        if not os.path.exists(gp): continue
        gt = np.asarray(nib.load(gp).dataobj).astype(np.int16)
        vv = float(np.prod(nib.load(gp).header.get_zooms()[:3]))/1000.0
        ssum = None
        for m in members:
            pr = np.load(probs_path(regime, TR[m], case, fold))["probabilities"].astype(np.float32)
            ssum = pr if ssum is None else ssum + pr
        pred = ssum.argmax(0).transpose(Tf).astype(np.int16)
        if pred.shape != gt.shape: continue
        for c in VIEW_FG[view]:
            p, g = (pred == c), (gt == c); d = p.sum()+g.sum()
            dsc.append(1.0 if d == 0 else 2*(p & g).sum()/d)
        if view == "SAX": pm[pid] = float((pred == SCAR).sum()*vv*DENS)
    pids = sorted(pm); pred = np.array([pm[p] for p in pids]); tru = np.array([tm[p] for p in pids])
    v, r = vpcc(pred, tru), best_rae(pred, tru)
    return float(np.mean(dsc)), v, r, 0.5*max(0, v)+0.5/(1+r)

CANDIDATES = [
    ("XL (size1)", ["XL"]),
    ("XL+DN (2)", ["XL", "DN"]),
    ("XL+DN+B7 (3)", ["XL", "DN", "B7"]),
    ("XL+DN+B7+HM (4)", ["XL", "DN", "B7", "HM"]),
    ("XL+DN+B7+ST+SP (5)", ["XL", "DN", "B7", "ST", "SP"]),
    ("XL+DN+L+ST+PC+SP (6=OOFbest)", ["XL", "DN", "L", "ST", "PC", "SP"]),
    ("XL+DN+B7+ST+PC+GA+SP (7)", ["XL", "DN", "B7", "ST", "PC", "GA", "SP"]),
    ("XL+DN+L+B7+PC (robust5)", ["XL", "DN", "L", "B7", "PC"]),
    ("XL+DN+L+B7 (clean4)", ["XL", "DN", "L", "B7"]),
    ("DN (mass best)", ["DN"]),
    ("ALL 11", list(TR.keys())),
]
cfO, cfV = cases_for("OOF"), cases_for("VAL")
tmO, tmV = true_mass("OOF"), true_mass("VAL")
print(f"OOF cases {len(cfO)} | 7-val cases {len(cfV)}\n")
print(f"{'combo':<32}{'OOF_DSC':>8}{'7v_DSC':>8}   {'OOF_mClin':>10}{'7v_mClin':>9}{'7v_vPCC':>8}")
for name, mem in CANDIDATES:
    do, vo, ro, co = eval_combo(mem, "OOF", cfO, tmO)
    dv, vv2, rv, cv = eval_combo(mem, "VAL", cfV, tmV)
    print(f"{name:<32}{do:>8.4f}{dv:>8.4f}   {co:>10.3f}{cv:>9.3f}{vv2:>8.3f}")
print("\n(7-val n=7, ~5 nonzero-scar -> vPCC/mass VERY noisy; DSC more stable. Confirmation, not selector.)")
