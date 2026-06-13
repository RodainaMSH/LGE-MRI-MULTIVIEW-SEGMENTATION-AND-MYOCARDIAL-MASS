# GAN scar-augmentation — data composition across the 5 folds

What the GAN lever actually feeds the segmenter, expanded patient → slice → class, for every fold.
*(Synthetic counts are the per-fold targets, ~300/fold split by real-scar proportion; the per-view quality gate
may drop a weak view to SAX and generation can fall slightly short — actual counts logged at BUILD_DONE.)*

## The hierarchy (so the units are clear)

- **1 patient** → has **3 views** (SAX, 2CH, 4CH)
- **1 view** → a stack of **2D slices** (SAX ~10, 2CH ~3, 4CH ~3) → so **1 patient ≈ 16 slices**
- **1 slice** → a 2D image whose *pixels* are labeled into **5 classes**: bg, LV-cavity, LV-myo, **scar (3)**, RV-cavity
- *"scar-slice"* = a slice with ≥1 scar pixel. Synthetic slices are **all** scar-bearing by construction.

The segmenter (nnU-Net 2D) trains **per slice**, so "32 patients" really means "512 real slices."

---

## Master summary — the 5 GANs / folds

Each fold = one independent experiment. The GAN is blind to that fold's 8 test patients **and** the 7-val (41–47),
so it trains on **32 patients**. Its ~300 synthetic slices feed **only that fold's** training.

| GAN / fold | held-out TEST patients (8) | trains on | real slices | real scar-slices | + synthetic | **train total** | scar after aug |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **GAN-0** | 1, 6, 8, 12, 19, 26, 31, 36 | 32 pts | 512 | 282 | ~300 | **~812** | 582 |
| **GAN-1** | 2, 7, 13, 17, 20, 27, 32, 37 | 32 pts | 512 | 302 | ~301 | **~813** | 603 |
| **GAN-2** | 3, 9, 14, 18, 21, 28, 33, 38 | 32 pts | 512 | 297 | ~301 | **~813** | 598 |
| **GAN-3** | 4, 10, 15, 22, 24, 29, 34, 39 | 32 pts | 512 | 292 | ~300 | **~812** | 592 |
| **GAN-4** | 5, 11, 16, 23, 25, 30, 35, 40 | 32 pts | 512 | 295 | ~300 | **~812** | 595 |
| **TOTAL** | all 40 covered once each | — | 2560 | 1468 | **~1502** | — | 2970 |

*(Every fold also excludes the 7-val 41–47 from the GAN — not shown, they never train on anything.)*
*"scar after aug" = real scar-slices + synthetic (all synthetic are scar) → roughly **doubled** per fold.*

---

## Per-fold breakdown — each GAN expanded by view

### Fold 0 · GAN-0 — blind to {1, 6, 8, 12, 19, 26, 31, 36}
| view | patients | ×slices/pt | real slices | of which scar | + synthetic | **view total** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SAX | 32 | ~10 | 320 | 153 | 163 | **483** |
| 2CH | 32 | ~3 | 96 | 65 | 69 | **165** |
| 4CH | 32 | ~3 | 96 | 64 | 68 | **164** |
| **TOTAL** | **32** | — | **512** | **282** | **300** | **812** |

### Fold 1 · GAN-1 — blind to {2, 7, 13, 17, 20, 27, 32, 37}
| view | patients | ×slices/pt | real slices | of which scar | + synthetic | **view total** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SAX | 32 | ~10 | 320 | 169 | 168 | **488** |
| 2CH | 32 | ~3 | 96 | 66 | 66 | **162** |
| 4CH | 32 | ~3 | 96 | 67 | 67 | **163** |
| **TOTAL** | **32** | — | **512** | **302** | **301** | **813** |

### Fold 2 · GAN-2 — blind to {3, 9, 14, 18, 21, 28, 33, 38}
| view | patients | ×slices/pt | real slices | of which scar | + synthetic | **view total** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SAX | 32 | ~10 | 320 | 160 | 162 | **482** |
| 2CH | 32 | ~3 | 96 | 70 | 71 | **167** |
| 4CH | 32 | ~3 | 96 | 67 | 68 | **164** |
| **TOTAL** | **32** | — | **512** | **297** | **301** | **813** |

### Fold 3 · GAN-3 — blind to {4, 10, 15, 22, 24, 29, 34, 39}
| view | patients | ×slices/pt | real slices | of which scar | + synthetic | **view total** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SAX | 32 | ~10 | 320 | 158 | 162 | **482** |
| 2CH | 32 | ~3 | 96 | 68 | 70 | **166** |
| 4CH | 32 | ~3 | 96 | 66 | 68 | **164** |
| **TOTAL** | **32** | — | **512** | **292** | **300** | **812** |

### Fold 4 · GAN-4 — blind to {5, 11, 16, 23, 25, 30, 35, 40}
| view | patients | ×slices/pt | real slices | of which scar | + synthetic | **view total** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SAX | 32 | ~10 | 320 | 164 | 167 | **487** |
| 2CH | 32 | ~3 | 96 | 67 | 68 | **164** |
| 4CH | 32 | ~3 | 96 | 64 | 65 | **161** |
| **TOTAL** | **32** | — | **512** | **295** | **300** | **812** |

---

## How the pieces stay leakage-clean

1. **GAN-k is blind to fold-k's 8 test patients + the 7-val (41–47)** → trains on 32 patients only.
2. **GAN-k's ~300 synthetic feed ONLY fold-k's training** (`build_clean_dataset.py --mode splits`:
   fold k train = real-fold-k-train + `SYNfk_*` only).
3. **Each patient is TESTED exactly once**, by the fold that held them out — whose model never saw them
   (real or synthetic). Patients overlap across the 5 training sets — that's normal k-fold CV, not a leak.
4. Routing is the linchpin: if all 1500 synthetic went into every fold, fold-0 would get GAN-1's synthetic
   (which saw fold-0's test patients) → leak. The 300-per-fold matching prevents it.

## The GAN model (same for all 5; only the training data differs)

- **pix2pix conditional GAN**: U-Net generator (5-channel one-hot label → 1-channel LGE, tanh) + 70×70 PatchGAN
  discriminator (6-channel [label|image]). LSGAN adversarial loss + L1 (λ=100), Adam 2e-4, 200 epochs, 256².
- Each synthetic slice: take a real label from the fold's 32 patients → **transplant scar to a new wall location**
  (a configuration not in the data) → GAN paints the matching LGE → resize to the view's real geometry.
- 5 GANs = **same architecture, different weights** (each trained on a different 32-patient subset).

## Bottom line

**5 GANs → ~1500 synthetic scar slices → used as 5 separate piles of ~300 → each pile trains only its own fold.**
Per fold the segmenter's diet goes 512 → ~812 slices, roughly **doubling scar exposure**. Scored on the real
held-out patients vs the DenseNet base (scar 0.3925 / Task2 0.704), gate **>+0.01 scar**.
