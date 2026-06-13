"""
Nested view->class DSC table (two-tier columns: each VIEW header spans an 'agg' sub-column
+ its class sub-columns). All 9 full-OOF models x 3 regimes. Emits markdown for LEADERBOARD.md.
agg: ① = patient-cell-pooled (matches Table A); ②③ = mean_dice over fg classes (matches Table A).
"""
import numpy as np, nibabel as nib
from collections import defaultdict
import sys; sys.path.insert(0,"/tmp/theirmetric")
from metrics import mean_dice
from nnunet.view_class_dsc import (oof_preds, our_dice, _sitk, their_perclass,
                                   ras_shared, OUR_FG, THEIR_NC, DATA, PREDV, SAX_IDS)
from nnunet.master_rows import (our_view_scar, their_view_scar, mass_oof, mass_val,
                                massscore, task2)  # aggregate metrics = Table A (import-safe now)
MODELS = [  # generates LEADERBOARD Table A2 (single models). For the Table-A ensemble row, temporarily
    # prepend ("Ensemble ⭐","ENS_best6__nnUNetPlans__2d","cand_ensbest6") (run ENS_best6 + cand_ensbest6 first).
    ("DenseNet-264",    "nnUNetTrainerSMP_densenet264__nnUNetPlans__2d","cand_dn264_ens_pp"),
    ("EffNetV2-XL",     "nnUNetTrainerSMP_effv2xl__nnUNetPlans__2d",    "cand_effv2xl_ens_pp"),
    ("EffNetV2-L",      "nnUNetTrainerSMP_effv2l__nnUNetPlans__2d",     "cand_effv2l_ens_pp"),
    ("EfficientNet-B7", "nnUNetTrainerSMP_effb7__nnUNetPlans__2d",      "cand_smpB7_ens_pp"),
    ("densenet121-RIN", "nnUNetTrainerRIN_densenet121__nnUNetPlans__2d","cand_dnrin_ens_pp"),
    ("ScarTversky",     "nnUNetTrainerScarTversky__nnUNetPlans__2d",    "cand_ensTv_pp"),
    ("PlainConv 100ep", "nnUNetTrainer_100epochs__nnUNetPlans__2d",     "cand_plainconv_pp"),
    ("Hiera-tiny",      "nnUNetTrainerHieraMedSAM2__nnUNetPlans__2d",   "cand_hiera_medsam2_ens_pp"),
    ("SAM2.1-large",    "nnUNetTrainerSAM2large__nnUNetPlans__2d",      "cand_sam2large_ens_pp"),
]
VC={"SAX":[1,2,3,4],"2CH":[1,2,3],"4CH":[1,2,3,4]}; CN={1:"LVcav",2:"LVmyo",3:"scar",4:"RVcav"}
RAS_OUR,RAS_THEIR=ras_shared()
def load(p): return np.asarray(nib.load(str(p)).dataobj).astype(int)

def compute(run,cand):
    pr=oof_preds(run)
    ours=defaultdict(lambda: defaultdict(list)); pool=defaultdict(list)
    for pid,vp in pr.items():
        for v,pp in vp.items():
            if v=="RAS": continue
            g=load(DATA/f"{v}_TR/anno/LGE_{v}_{pid:03d}.nii.gz"); p=load(pp)
            if g.shape!=p.shape: continue
            for c,val in our_dice(p,g,OUR_FG[v]).items(): ours[v][c].append(val); pool[v].append(val)
    def their(val):
        out=defaultdict(dict)
        for v in ["SAX","2CH","4CH"]:
            nc=THEIR_NC[v]; sp,sg=[],[]
            if val:
                for pid in SAX_IDS:
                    pf=PREDV/cand/f"{v}_{pid:03d}.nii.gz"; gf=DATA/f"{v}_VAL/anno/LGE_{v}_{pid:03d}.nii.gz"
                    if pf.exists() and gf.exists():
                        a,b=_sitk(pf),_sitk(gf)
                        if a.shape==b.shape:
                            for d in range(b.shape[0]): sp.append(a[d]); sg.append(b[d])
            else:
                for pid in sorted(pr):
                    if v not in pr[pid]: continue
                    a,b=_sitk(pr[pid][v]),_sitk(DATA/f"{v}_TR/anno/LGE_{v}_{pid:03d}.nii.gz")
                    if a.shape==b.shape:
                        for d in range(b.shape[0]): sp.append(a[d]); sg.append(b[d])
            per=their_perclass(sp,sg,nc)
            for c in range(1,nc): out[v][c]=per[c]
        return out
    two=their(False); three=their(True)
    def cell(reg,v,c):
        if reg=="①": return float(np.mean(ours[v][c])) if ours[v][c] else float('nan')
        return (two if reg=="②" else three)[v].get(c,float('nan'))
    def agg(reg,v):
        if reg=="①": return float(np.mean(pool[v])) if pool[v] else float('nan')
        d=(two if reg=="②" else three)[v]; return mean_dice([0]+[d[c] for c in sorted(d)]) if d else float('nan')
    return cell,agg

DESC = {  # arch · pretraining · params (mirrors Table A)
    "Ensemble ⭐":     "soft-avg 6 archs: XL+DN+V2L+ScarTv+PlainConv+SPADE-aug",
    "DenseNet-264":    "smp-UNet · in1k PaddleClas · ~44M",
    "EffNetV2-XL":     "smp-UNet · in21k→1k · ~209M",
    "EffNetV2-L":      "smp-UNet · in21k→1k · ~120M",
    "EfficientNet-B7": "smp-UNet · in1k · ~66M",
    "densenet121-RIN": "smp-UNet · RadImageNet medical · ~8M",
    "ScarTversky":     "PlainConv+Focal-Tversky · scratch · ~30M",
    "PlainConv 100ep": "nnU-Net PlainConvUNet · scratch · ~30M",
    "Hiera-tiny":      "smp-UNet · MedSAM2 trunk · ~27M",
    "SAM2.1-large":    "smp-UNet · SAM2.1 Hiera-large · ~214M",
}
def f(x): return "—" if (x!=x) else f"{x:.3f}"
def f2(x): return "—" if (x!=x) else f"{x:.2f}"
# HTML multi-level header. Per view: numbered classes then DSC. Then Overall DSC + Table-A metrics.
C='align="center" style="text-align:center"'  # both (VSCode ignored bare align)
print('<table>')
print('<thead>')
print(f'<tr><th rowspan="2">Model</th><th rowspan="2" {C}>Reg</th>'
      f'<th colspan="5" {C}>SAX</th><th colspan="4" {C}>2CH</th>'
      f'<th colspan="5" {C}>4CH</th><th {C}>RAS</th>'
      f'<th rowspan="2" {C}><b>Overall<br>DSC</b></th>'
      f'<th rowspan="2" {C}>scar</th><th rowspan="2" {C}>vPCC</th><th rowspan="2" {C}>RAE↓</th>'
      f'<th rowspan="2" {C}>a</th><th rowspan="2" {C}>Mass</th><th rowspan="2" {C}><b>Task2</b></th></tr>')
print(f'<tr><th {C}>① LVcav</th><th {C}>② LVmyo</th><th {C}>③ scar</th><th {C}>④ RVcav</th><th {C}><i>DSC</i></th>'
      f'<th {C}>① LVcav</th><th {C}>② LVmyo</th><th {C}>③ scar</th><th {C}><i>DSC</i></th>'
      f'<th {C}>① LVcav</th><th {C}>② LVmyo</th><th {C}>③ scar</th><th {C}>④ RVcav</th><th {C}><i>DSC</i></th>'
      f'<th {C}>① RA</th></tr>')
print('</thead>')
print('<tbody>')
for name,run,cand in MODELS:
    cell,agg=compute(run,cand)
    # aggregate metrics (= Table A): scar per regime + OOF/val mass (vPCC/RAE/a/Mass)
    _,scar1 = our_view_scar(run); _,scar2 = their_view_scar(run=run)
    vp,rae,a = mass_oof(run); mO = massscore(vp,rae)
    scar3=vpv=raev=m3=float('nan')
    if cand:
        _,scar3 = their_view_scar(cand=cand); vpv,raev = mass_val(cand,a); m3 = massscore(vpv,raev)
    for i,reg in enumerate(["①","②","③"]):
        ra = RAS_OUR if reg=="①" else RAS_THEIR
        sx,c2,c4 = agg(reg,"SAX"),agg(reg,"2CH"),agg(reg,"4CH")
        vals=[cell(reg,"SAX",c) for c in VC["SAX"]]+[sx]
        vals+=[cell(reg,"2CH",c) for c in VC["2CH"]]+[c2]
        vals+=[cell(reg,"4CH",c) for c in VC["4CH"]]+[c4]
        vals+=[ra]
        overall=float(np.nanmean([sx,c2,c4,ra]))   # 4-view mean = Table A DSC
        if reg=="③": sc,pv,re,al,ms,t2 = scar3,vpv,raev,a,m3,(task2(overall,vpv,raev) if cand else float('nan'))
        else:        sc,pv,re,al,ms,t2 = (scar1 if reg=="①" else scar2),vp,rae,a,mO,task2(overall,vp,rae)
        lbl={"①":"① Ours","②":"② Paper","③":"③ 7-val"}[reg]
        tds="".join(f'<td {C}>{f(v)}</td>' for v in vals)
        tds+=f'<td {C}><b>{f(overall)}</b></td>'
        tds+=(f'<td {C}>{f(sc)}</td><td {C}>{f(pv)}</td><td {C}>{f(re)}</td>'
              f'<td {C}>{f2(al)}</td><td {C}>{f(ms)}</td><td {C}><b>{f(t2)}</b></td>')
        head=f'<td rowspan="3"><b>{name}</b><br><i>{DESC[name]}</i></td>' if i==0 else ''
        print(f'<tr>{head}<td {C}>{lbl}</td>{tds}</tr>')
print('</tbody>')
print('</table>')
