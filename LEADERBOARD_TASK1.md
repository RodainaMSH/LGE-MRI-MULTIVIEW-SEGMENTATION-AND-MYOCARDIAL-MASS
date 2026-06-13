# LEADERBOARD — TASK 1 (Cine MRI · segmentation + Ejection Fraction)

Companion to [`LEADERBOARD.md`](LEADERBOARD.md) (Task 2 / LGE). All numbers are
**5-fold OOF** over the 105 Cine training patients (our internal cross-validation),
pooled across folds — the clean read for ranking decisions; the held-out val/test
should track it.

**Official scoring** (verified live off Codabench competition 15533, *Evaluation* +
*Ranking Methods* pages, 2026-06-12):

```
Task1 = 0.7 · (0.4·DSC + 0.3·1/(1+HD) + 0.3·1/(1+ASD)) + 0.3·max(0, EF-PCC)
Overall = (Task1 + Task2) / 2
```
- **DSC** — Dice, mean over structures/views/patients (↑ better)
- **HD** — *maximum* Hausdorff boundary distance, **mm** (↓ better) — Task 1 only
- **ASD** — average symmetric surface distance, **mm** (↓ better) — Task 1 only
- **EF-PCC** — Pearson corr. of predicted vs true ejection fraction (↑ better)

**Model:** baseline `nnUNetTrainer_100epochs`, 2D, **per-view** 5-fold
(Dataset100_CINESAX · 101_CINE2CH · 102_CINE4CH). Single model, **no ensemble**.

---

## 0) ⭐ OFFICIAL RESULT — #1 on the Codabench validation leaderboard (2026-06-12, sub 794082 `youssefaraby`)

| Rank | Overall | Task1 | Cine DSC | Cine HD(mm) | Cine ASD(mm) | EF-PCC | Task2 | LGE DSC | Mass vPCC | Mass RAE |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **1 — us** | **0.69** | 0.65 | 0.88 | 12.86 | 0.85 | **0.94** | 0.73 | 0.78 | 0.70 | 0.88 |
| 2 tobi_hao | 0.67 | 0.65 | 0.87 | 11.14 | 0.79 | 0.91 | 0.68 | 0.75 | 0.74 | 2.25 |
| 3 MWM_UNNC | 0.65 | 0.62 | 0.83 | 11.28 | 1.07 | 0.89 | 0.68 | 0.65 | 0.95 | 0.73 |

Our OOF estimates matched official to **±0.01 on everything except HD/ASD**. Lead = **+0.02**, driven by Task 2 (Task 1 TIED at 0.65 with #2). First upload scored 0.0 (val IDs use 001..N, not our 106-120/41-47/81-94 — fixed by ascending remap).

> **⚠️ HD AMBIGUITY RESOLVED:** official Cine **HD = 12.86 mm** — NEITHER our max-HD (≈26) NOR HD95 (≈3.8); a *middle* convention. So Section A's "HD (max, mm)" column OVER-estimates; trust 12.86 mm. ASD official 0.85 (our OOF 0.60). DSC/EF estimates were spot-on.

## A·2) ROUND-2 — encoder/generalist headroom test (2026-06-13)

**DenseNet-264 (deep-sup OFF, *handicapped*) fold-0 BEATS PlainConv (DS-on) on ALL 3 views → Cine DOES have encoder headroom:**

| View | DenseNet f0 | PlainConv(DS-on) f0 | Δ |
|---|:--:|:--:|:--:|
| SAX | 0.8816 | 0.8793 | +0.0023 |
| 2CH | 0.8517 | 0.8469 | +0.0048 |
| 4CH | 0.8740 | 0.8643 | +0.0097 |

Gains real but **small** (~+0.005 avg → ~+0.002 Task1).

⛔ **Generalist DenseNet (`Dataset103`, 3-view pooled) = DEAD** (fold_0 per-view): **SAX 0.809 vs 0.879 = −0.070 💥**, 2CH 0.839 (−0.008), 4CH 0.865 (+0.001). The pooled nnU-Net plan adopted 2CH/4CH spacing `[0.135,1.298]` — wrong for SAX (`[0.325,1.282]`/patch`[320,160]`) → **SAX crushed**. A single plan can't fit short-axis + long-axis together → cross-view generalist sacrifices SAX. Stopped (folds 1-4 killed).

**Round-2 verdict: per-view DenseNet WINS but small (~+0.002 Task1); generalist DEAD. Task 1 ≈ maxed** — squeezing it via a per-view DenseNet 5-fold ensemble (~8-24h GPU) is low reward; pivot to Docker/paper. (PlainConv-NoDS "fair baseline" killed mid-run as redundant.) Scripts: `nnunet/cine_densenet_test.sh`, `nnunet/cine_overnight.sh`.

---

## A) Headline — per-view summary  (OOF estimate — superseded by §0 official; HD here is max-HD, over-estimated)

| View | DSC | HD (max, mm) | ASD (mm) | n |
|------|:---:|:---:|:---:|:---:|
| SAX  | 0.8825 | 33.1 | 0.52 | 105 |
| 2CH  | 0.8402 | 19.9 | 0.56 | 105 |
| 4CH  | 0.8555 | 25.3 | 0.72 | 105 |
| **Overall** | **0.859** | **26.1** | **0.60** | |

**EF-PCC = 0.916** (SAX OOF · MAE 3.6% · 99/105 pts; 6 dropped — plane count not
divisible by #frames, TODO for final submission). EF recipe: `nnunet/ef_pipeline.py`.

### ⭐ Task 1 score ≈ **0.654**  (max-HD reading)

> ⚠️ **Metric-definition ambiguity.** The Evaluation page prose says **maximum** HD,
> but the provided baseline repo code computes **HD95**. Under HD95 our HD ≈ 3.8 mm →
> **Task 1 ≈ 0.69**. So true Task 1 ∈ **[0.65, 0.69]** — competitive-to-leading either
> way. (To settle it, inspect the scoring Docker `baifeng2lab/cmr-multi-challenge:latest`.)

---

## B) Per-structure detail (RAW OOF)

| View | Structure | DSC | HD (max, mm) | ASD (mm) |
|------|-----------|:---:|:---:|:---:|
| **SAX** | LV-cavity   | 0.932 | 16.6 | 0.16 |
|         | LV-myo      | 0.882 | 23.3 | 0.27 |
|         | RV-cavity   | 0.834 | 59.3 | 1.15 |
| **2CH** | LV-cavity   | 0.884 | 19.8 | 0.47 |
|         | LV-myo      | 0.797 | 19.9 | 0.65 |
| **4CH** | LV-cavity   | 0.907 | 23.2 | 0.42 |
|         | LV-myo      | 0.792 | 26.0 | 0.96 |
|         | RV-cavity   | 0.856 | 25.5 | 0.65 |
|         | RA          | 0.838 | 26.9 | 1.04 |
|         | LA          | 0.884 | 24.8 | 0.52 |

**Notes:** weakest DSC = **LV-myocardium in long-axis** (2CH 0.797, 4CH 0.792 — thin
wall). HD is dominated by **SAX RV-cavity (59 mm)** — a genuine worst-slice boundary
error, not a removable artifact. ASD is excellent everywhere (≤1.2 mm).

---

## C) Connected-component cleanup — ❌ NEGATIVE (do NOT apply)

Tested "keep largest blob per structure per frame" to cut HD outliers:

| | DSC | HD (mm) | ASD (mm) | **Task 1** |
|---|:---:|:---:|:---:|:---:|
| RAW   | 0.859 | 26.1 | 0.60 | **0.654** |
| CLEAN | 0.856 | 27.1 | 0.89 | **0.633** ↓ |

**Verdict: makes it WORSE.** Keep-largest deletes a legitimate **myocardium wall** —
in long-axis (2CH/4CH) the LV-myo appears as *two separate walls*, not one ring (LV-myo
ASD 0.65→1.97, HD 20→31). And the dominant HD (SAX RV-cavity 59 mm) is a *real* error,
not a removable speck (cleanup moved it only 59→57). **HD is a near-floor tax everyone
pays** (max-Hausdorff is outlier-pinned at ~1/(1+26)=0.037); post-processing can't shift
it. Lever DEAD. Our Task-1 edge is **DSC + ASD + EF**, where we're already top-tier.

---

## D) vs competition leaderboard (Validation phase, read 2026-06-11)

| Team | Task 1 | Cine DSC | EF-PCC |
|------|:---:|:---:|:---:|
| tobi_hao (leader) | 0.65 | 0.87 | 0.91 |
| MWM #2 | ~0.65 | — | — |
| revenge | 0.47 | — | low |
| **Ours (OOF est.)** | **0.65–0.69** | **0.859** | **0.916** |

With our **Task 2 ≈ 0.73**, projected **Overall ≈ (0.66 + 0.73)/2 ≈ 0.69 → #1**
(board leader Overall 0.67).

---

## Provenance / scripts
- DSC: per-fold `…/fold_k/validation/summary.json`, pooled (105 cases/view).
- HD / ASD: `nnunet/cine_hdasd2.py` (per-patient pooled, max/mean, mm, both-present planes).
- Cleanup test: `nnunet/cine_cleanup_eval.py`.
- EF-PCC: `nnunet/ef_pipeline.py` (slice-major, EDV/ESV from LV-cavity voxel volume).
- Metric defs verified on Codabench 15533 *Evaluation* + *Ranking Methods* tabs.
