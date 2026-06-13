"""
Official-aligned HD + ASD for Cine OOF, per Codabench Evaluation page:
  HD  = MAXIMUM symmetric surface distance (mm)   [not HD95]
  ASD = MEAN  symmetric surface distance (mm)
  "per structure, averaged across structures/views/patients".
Per-patient pooling: for each (patient, class), pool all symmetric surface
distances over the planes where the structure is present in BOTH pred & GT;
HD = max(pool) mm, ASD = mean(pool) mm. (One-sided/empty planes are presence
errors captured by DSC; excluded from the distance pool.) mm via per-case
in-plane spacing. Task1 transform = 1/(1+x).
"""
import glob, numpy as np, nibabel as nib
from scipy.ndimage import distance_transform_edt

RES, RAW = "nnunet/nnUNet_results", "nnunet/nnUNet_raw"
VIEWS = {100: ("SAX", {1: "LVmyo", 2: "LVcav", 3: "RVcav"}),
         101: ("2CH", {1: "LVcav", 2: "LVmyo"}),
         102: ("4CH", {1: "LVcav", 2: "LVmyo", 3: "RVcav", 4: "RA", 5: "LA"})}

view_hd, view_asd = {}, {}
for d, (vname, names) in VIEWS.items():
    run = glob.glob(f"{RES}/Dataset{d}_*/nnUNetTrainer_100epochs__nnUNetPlans__2d")[0]
    lbl = glob.glob(f"{RAW}/Dataset{d}_*/labelsTr")[0]
    fg = list(names)
    cls_hd = {c: [] for c in fg}; cls_asd = {c: [] for c in fg}
    cases = [pf for k in range(5) for pf in sorted(glob.glob(f"{run}/fold_{k}/validation/*.nii.gz"))]
    for pf in cases:
        case = pf.split("/")[-1][:-7]
        g = nib.load(f"{lbl}/{case}.nii.gz"); ga = np.asarray(g.dataobj).astype(np.int16)
        pa = np.asarray(nib.load(pf).dataobj).astype(np.int16)
        if pa.shape != ga.shape: continue
        sx, sy = g.header.get_zooms()[:2]; sp = float((sx + sy) / 2)   # mm/pixel
        pp, gg = np.moveaxis(pa, -1, 0), np.moveaxis(ga, -1, 0)
        for c in fg:
            pool = []
            for k in range(pp.shape[0]):
                pb, tb = (pp[k] == c), (gg[k] == c)
                if not pb.any() or not tb.any(): continue          # need both present
                dpg = distance_transform_edt(~tb) * sp
                dgp = distance_transform_edt(~pb) * sp
                pool.append(dpg[pb]); pool.append(dgp[tb])
            if pool:
                alld = np.concatenate(pool)
                cls_hd[c].append(float(alld.max())); cls_asd[c].append(float(alld.mean()))
    print(f"\n===== {vname} (Dataset{d}) — per-patient pooled, mm =====")
    print(f"  {'class':<8}{'HD(max,mm)':>12}{'ASD(mm)':>10}")
    vh, va = [], []
    for c in fg:
        h = np.mean(cls_hd[c]) if cls_hd[c] else float('nan')
        a = np.mean(cls_asd[c]) if cls_asd[c] else float('nan')
        vh.append(h); va.append(a)
        print(f"  {names[c]:<8}{h:>12.2f}{a:>10.2f}  (n={len(cls_hd[c])})")
    view_hd[vname] = np.nanmean(vh); view_asd[vname] = np.nanmean(va)
    print(f"  {'MEAN':<8}{view_hd[vname]:>12.2f}{view_asd[vname]:>10.2f}")

HD = float(np.mean(list(view_hd.values()))); ASD = float(np.mean(list(view_asd.values())))
DSC, EF = 0.859, 0.916
task1 = 0.7 * (0.4 * DSC + 0.3 / (1 + HD) + 0.3 / (1 + ASD)) + 0.3 * max(0, EF)
print(f"\n>>> OVERALL (mean of views): HD = {HD:.2f} mm | ASD = {ASD:.2f} mm")
print(f">>> with DSC {DSC}, EF-PCC {EF}:")
print(f">>>   1/(1+HD)={1/(1+HD):.3f}  1/(1+ASD)={1/(1+ASD):.3f}")
print(f">>>   TASK1 = 0.7*(0.4*{DSC} + 0.3*{1/(1+HD):.3f} + 0.3*{1/(1+ASD):.3f}) + 0.3*{EF} = {task1:.4f}")
print("\nDONE_HDASD2")
