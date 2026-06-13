"""
Assemble Task 1 (Cine) submission from the 5-fold val predictions in /tmp/cval/out_{view}:
  - copy/clamp masks -> SUBMISSION/task1_cine/{SAX,2CH,4CH}/CINE_{view}_{pid}.nii.gz
  - EF from SAX (slice-major, LVcav=label 2) -> SUBMISSION/task1_cine/ef_predictions.json
  - self-validate EF-PCC vs the true val LVEF (dataset_valid.xlsx) as a sanity check.
"""
import numpy as np, nibabel as nib, json, glob, os, pandas as pd
ROOT="/home/youssef/projects/research/CMR-MULTI"; SUB=f"{ROOT}/SUBMISSION"
VALID={"SAX":{0,1,2,3},"2CH":{0,1,2},"4CH":{0,1,2,3,4,5}}
LVCAV=2  # CINE SAX native: 1 LV_Myo, 2 LV_Cavity, 3 RV_Cavity

def clamp(a,valid):
    a=np.round(np.asarray(a)).astype(np.int16); bad=~np.isin(a,list(valid))
    if bad.any(): a=a.copy(); a[bad]=0
    return a

# 1) masks
for v in ("SAX","2CH","4CH"):
    od=f"{SUB}/task1_cine/{v}"; os.makedirs(od,exist_ok=True)
    for f in sorted(glob.glob(f"/tmp/cval/out_{v}/CINE_{v}_*.nii.gz")):
        img=nib.load(f); a=clamp(img.dataobj,VALID[v])
        nib.save(nib.Nifti1Image(a,img.affine,img.header),f"{od}/{os.path.basename(f)}")
    print(f"  {v}: {len(glob.glob(f'{od}/*.nii.gz'))} masks")

# 2) EF
frames=json.load(open(f"{ROOT}/CINE_MULTI/id_slice_info_valid.json"))   # "106"->n_frames
df=pd.read_excel(f"{ROOT}/CINE_MULTI/dataset_valid.xlsx",sheet_name="SAX")
true={int(r.patient_id):float(r.LVEF)*100.0 for r in df.itertuples()}    # fraction -> %
def ef_slice_major(cav,nfr):
    n=len(cav); nsl=n//nfr
    if nsl*nfr!=n: return None
    ft=np.array(cav).reshape(nsl,nfr).sum(0); edv,esv=ft.max(),ft.min()
    return float(100.0*(edv-esv)/edv) if edv>0 else 0.0
ef={}; checked=[]; dropped=[]
for f in sorted(glob.glob(f"{SUB}/task1_cine/SAX/CINE_SAX_*.nii.gz")):
    pid=int(os.path.basename(f).split("_")[2][:3]); key=os.path.basename(f)[:-7]
    pr=np.asarray(nib.load(f).dataobj).astype(np.int16)
    cav=(pr==LVCAV).reshape(-1,pr.shape[2]).sum(0)
    e=ef_slice_major(cav,frames.get(str(pid),0))
    if e is None: dropped.append(key); continue
    ef[key]=round(e,2)
    if pid in true: checked.append((pid,true[pid],e))
# fallback for dropouts: cohort median
if dropped:
    med=float(np.median(list(ef.values()))) if ef else 50.0
    for k in dropped: ef[k]=round(med,2)
    print(f"  EF dropouts (planes%frames!=0): {dropped} -> filled median {med:.1f}")
json.dump(ef,open(f"{SUB}/task1_cine/ef_predictions.json","w"),indent=1)
print(f"  ef_predictions.json: {len(ef)} patients")
if checked:
    t=np.array([c[1] for c in checked]); p=np.array([c[2] for c in checked])
    pcc=float(np.corrcoef(t,p)[0,1]); mae=float(np.mean(np.abs(t-p)))
    print(f"  *** EF self-check on {len(checked)} val pts: EF-PCC {pcc:+.3f}  MAE {mae:.1f}% ***")
    print("  sample (pid true/pred): "+", ".join(f"{c[0]}:{c[1]:.0f}/{c[2]:.0f}" for c in checked[:8]))
print("TASK1_BUILD_DONE")
