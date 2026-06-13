# Training pipeline — how the data becomes a score

The journey from the assembled slices (see [gan.md](gan.md)) to a decision-grade number.
Done **5× independently** (one per fold), then stitched into one out-of-fold (OOF) result.

```
  real CMR slices ───────────────┐
   (512 / fold, 40 patients)     │
                                 ├──▶  ~812 slices/fold ──▶ ① nnU-Net RECIPE ──▶ ② U-Net RECIPE ──▶ ③ OOF INFERENCE ──▶ ④ SCORE
  pix2pix GAN ──▶ synthetic ─────┘                          preprocessing +       backbone:          each fold predicts     scar Dice
   scar slices (300 / fold)                                 aug + optim/LR        DenseNet-264+U-Net  its 8 held-out reals   + Task2 (vPCC,RAE)
```

---

## ① The nnU-Net recipe — the framework  *(preprocessing + augmentation + optimizer / LR)*

**nnU-Net** (the *framework*) wraps the network and owns everything around it — preprocessing and the whole
training loop. It's all auto-configured from the plan and held **identical to the base run**; only the network (②)
and the data differ.

**Preprocessing** *(once, every slice)*:
1. **Resampled** to the plan's spacing `[0.647, 0.691]` mm → a pixel means the same physical size everywhere.
2. **Z-score normalized per-image** (mean 0, std 1).

> **Why the z-score matters for the GAN:** it re-scales the GAN's `[-1,1]` output onto the *same statistical
> footing* as real slices. Absolute intensity is erased; only the **scar-vs-myocardium contrast** survives —
> what the quality gate validated. The segmenter can't tell a synthetic slice from a real one by its intensity
> range. *(Plan is transferred from Dataset000 via `move_plans`, so patch/spacing/norm are identical to the base —
> the only difference vs base is the +300 synthetic slices.)*

**The full nnU-Net recipe** *(preprocessing + training loop):*

| stage | setting |
|---|---|
| preprocessing | resample [0.647, 0.691] → per-image z-score *(above)* |
| **augmentation — rotation** | **±180°** full range *(verified `nnUNetTrainer.py:484`; full because the 224² patch is square/isotropic)* |
| augmentation — scaling / mirror | 0.7–1.4× / both axes |
| augmentation — intensity | Gaussian noise, blur, brightness, contrast, simulate-low-res, gamma *(elastic **off**)* |
| optimizer | **SGD**, momentum 0.99, Nesterov |
| LR scheduler | **PolyLR** (polynomial decay, exponent 0.9), initial LR **0.01** |
| weight decay | 3e-5 |
| loss | **Dice + CrossEntropy** |
| **foreground oversampling** | **33%** of patches forced to contain a foreground class |
| schedule | 100 epochs × 250 iters, batch 17, **5 independent folds** |

---

## ② The U-Net recipe — the backbone  *(encoder / decoder / skip-connections)*

The network that nnU-Net's recipe (①) drives. For fold *k* it trains on **only** that fold's ~812 slices
(512 real + 300 synthetic), producing **5 independent models** (one per fold).

A U-Net is an **encoder → decoder with skip-connections**. We keep that entire shape and swap only the encoder slot:

| U-Net part | what it does | in our setup |
|---|---|---|
| **encoder** (contracting path) | downsamples, extracts features fine→coarse ("what's in here") | **← the only swapped slot:** DenseNet-264, ImageNet-1k pretrained (~30.6M), instead of from-scratch convs |
| bottleneck | most compressed, most abstract features | part of the encoder backbone |
| **decoder** (expanding path) | upsamples back to full resolution → per-pixel mask ("where exactly") | **kept** — smp U-Net decoder, channels 256→128→64→32→16 |
| **skip-connections** | hand fine detail from each encoder level to the matching decoder level so boundaries survive downsampling | **kept** |

> **Framework note:** the nnU-Net plan lists `PlainConvUNet`, but our trainer **overrides** `build_network_architecture`
> to build an **smp U-Net** with the DenseNet-264 encoder in that slot. Input = 1-ch 224×224 (RGB→1ch by summing
> conv0 weights, verified at load); output = single 5-class map, **deep supervision OFF**.

> **Where the synthetic earns its keep:** the 33% foreground-oversampler (①) now draws from 300 extra **scar**
> slices, so the model spends a third of its time on scar — in *more locations* than the 32 real patients ever
> showed. The bet: a detector trained on more scar configurations generalizes to unseen patients.

Output: **5 trained checkpoints** (fold_0 … fold_4).

---

## ③ Inference — two paths  *(OOF = measure · ensemble = deploy)*

The 5 fold-models get used **two different ways**, and it matters which:

### ③a · OOF inference — the *measurement* (what we gate the GAN on)

Fold-*k*'s model predicts on **only its 8 held-out real patients** — sliding over each slice, emitting a 5-class
label per pixel. The 5 folds partition all 40 patients, so **every patient is predicted exactly once, by the one
model that never saw it** (real *or* synthetic). Stitching the 5 → the OOF set over all 40. **No ensembling here**
— ensembling would let a model help grade a patient it trained on. This is the honest 40-patient score in ④.

### ③b · Ensemble inference — the *deliverable* (unseen data: the 7-val / test set)

On data *no* fold trained on, all **5 fold-models run together** and are combined:

> **Ensembling method = softmax (probability) averaging — "soft voting".** Each of the 5 models outputs per-pixel
> class *probabilities*; nnU-Net **averages the 5 probability maps**, then takes the per-pixel **argmax**. It is *not*
> a majority vote on hard labels — it averages the soft probabilities first, then decides (`nnUNetv2_predict -f 0 1 2 3 4`).
> Then nnU-Net **post-processing** (largest-connected-component cleanup, from the OOF `postprocessing.pkl`) → `cand_*_ens_pp`.

So the ensemble draws on **all 40 patients' knowledge** (the union of the 5 folds) — which is why the deployed/7-val
predictor is stronger than any single 32-patient fold. *(Exception: the RAS view uses a separate fold_0-only expert,
not the ensemble.)*

---

## ④ Score  *(scar Dice + the mass terms)*

**Top line:**

```
Task2 = 0.7 · DSC  +  0.3 · MassScore
```

- **DSC** = scar Dice, our per-volume skip-empty metric (`2·|P∩G| / (|P|+|G|)`, patients with no GT scar skipped).
  This is the 70% — the primary target the GAN is trying to move.

**Mass term (the 30%) — this is where vPCC and RAE live:**

```
MassScore = 0.5 · max(0, vPCC)  +  0.5 / (1 + RAE)
```

Scar **mass** per patient = `scar_voxel_count × voxel_volume_mL × 1.05 g/mL`, **SAX-only** (the scored view).

| term | formula | what it measures | behavior |
|---|---|---|---|
| **vPCC** | `corrcoef(pred_mass, true_mass)[0,1]` across patients | do **bigger true scars → bigger predicted scars**? (ranking/correlation) | **scale-invariant** — calibration can't change it; only the segmentation's mass *ordering* does |
| **RAE** | `mean( |pred − true| / true )` over scar-bearing patients | **absolute mass accuracy** per patient (relative error) | punished by over- or under-segmentation; **a per-mass scale `a` can be calibrated to minimize it** (vPCC stays fixed) |

- **vPCC** rewards getting the *trend* right (patient A has more scar than B). Range −1…1; only the positive part counts (`max(0,·)`).
- **RAE** rewards getting the *amount* right. RAE = 0 → perfect → contributes the full 0.5; RAE = 1 → contributes 0.25.
- **Calibration:** an intercept is wrong here (it inflates ~zero-scar patients and blows up RAE), so we fit a single
  multiplicative scale `a` on the OOF to minimize RAE, then apply it to held-out data. vPCC is unaffected by `a`.

**The verdict (gate):** compare GAN-aug OOF vs the **DenseNet base**:

| model | scar DSC | vPCC | RAE | Task2 ① |
|---|---|---|---|---|
| DenseNet-264 **base** | **0.3925** | 0.650 | *(base run)* | **0.704** |
| DenseNet-264 **+ GAN aug** | *(this run)* | *(this run)* | *(this run)* | *(this run)* |

**Pass:** scar DSC **> +0.01** (promising) / **+0.02** (decisive), confirmed by **paired per-case win/loss** so one
lucky patient can't fake it. Win → it becomes our scar+mass model. Marginal (≤+0.005) → not worth the complexity.
Loss → honest negative, GAN lever closed.

---

## The full arc (one line)

**32 patients → 512 real slices → +300 synthetic scar → DenseNet trains 25k steps with ⅓ attention forced onto
scar → predicts on the 8 it never saw → ×5 folds → one OOF (scar Dice, vPCC, RAE) held against the base.**
If extra scar variety lifted recall, it shows here; if the model just learned GAN texture, that shows here too.
