import numpy as np, nibabel as nib, json, glob, os, pandas as pd
LVCAV=2  # SAX native: 1 LV_Myo, 2 LV_Cavity, 3 RV_Cavity
run="nnunet/nnUNet_results/Dataset100_CINESAX/nnUNetTrainer_100epochs__nnUNetPlans__2d"
frames=json.load(open("CINE_MULTI/sax_slice_info.json"))   # pid(3-digit)->n_frames
# true LVEF (strip %)
df=pd.read_excel("CINE_MULTI/dataset_train.xlsx",sheet_name="SAX")
true={int(r.patient_id):float(str(r.LVEF).replace('%','')) for r in df.itertuples()}
# gather OOF preds (each fold's validation)
preds={}
for p in glob.glob(f"{run}/fold_*/validation/SAX_*.nii.gz"):
    pid=int(os.path.basename(p).split("_")[1][:3]); preds[pid]=p
def ef_from(planes_lvcav_count, nfr, order):
    n=len(planes_lvcav_count); nsl=n//nfr
    if nsl*nfr!=n: return None
    a=np.array(planes_lvcav_count)
    if order=="slice_major":  # [s0f0..s0fF, s1f0..]: frame totals = sum over slices = reshape(nsl,nfr).sum(0)
        ft=a.reshape(nsl,nfr).sum(0)
    else:                      # frame_major [f0s0..f0sS,...]
        ft=a.reshape(nfr,nsl).sum(1)
    edv,esv=ft.max(),ft.min()
    return 100.0*(edv-esv)/edv if edv>0 else 0.0
rows=[]
for pid,pf in sorted(preds.items()):
    if pid not in true or str(pid).zfill(3) not in frames: continue
    pr=np.asarray(nib.load(pf).dataobj).astype(np.int16)  # (H,W,planes)
    cav=(pr==LVCAV).reshape(-1,pr.shape[2]).sum(0)        # LV-cav voxels per plane
    nfr=frames[str(pid).zfill(3)]
    rows.append((pid, true[pid], ef_from(cav,nfr,"slice_major"), ef_from(cav,nfr,"frame_major")))
rows=[r for r in rows if r[2] is not None and r[3] is not None]
t=np.array([r[1] for r in rows]); sm=np.array([r[2] for r in rows]); fm=np.array([r[3] for r in rows])
def pcc(a,b): return float(np.corrcoef(a,b)[0,1])
def mae(a,b): return float(np.mean(np.abs(a-b)))
print(f"n patients EF'd: {len(rows)} / {len(true)}")
print(f"  slice_major reshape: EF-PCC {pcc(sm,t):+.3f}   MAE {mae(sm,t):.1f}%")
print(f"  frame_major reshape: EF-PCC {pcc(fm,t):+.3f}   MAE {mae(fm,t):.1f}%")
best="slice_major" if pcc(sm,t)>=pcc(fm,t) else "frame_major"; be=sm if best=="slice_major" else fm
print(f"  -> BEST ordering = {best}: EF-PCC {pcc(be,t):+.3f}  (leaderboard winners ~0.89-0.91)")
print(f"  sample (pid true_EF pred_EF): "+", ".join(f"{r[0]}:{r[1]:.0f}/{(r[2] if best=='slice_major' else r[3]):.0f}" for r in rows[:6]))
