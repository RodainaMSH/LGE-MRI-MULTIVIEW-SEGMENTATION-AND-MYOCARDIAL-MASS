# U-Net Cardiac Cine MRI Segmentation — Multi-View

A PyTorch implementation of U-Net for automated cardiac segmentation on multi-view Cine MRI data. A single notebook handles training, metric logging, chart generation, evaluation, and prediction visualisation across three cardiac views: Short Axis (SAX), 2-Chamber (2CH), and 4-Chamber (4CH).

---

## Project Structure

```
├── Unet_Cine_multi.ipynb        # Single notebook — train → log → plot → evaluate → visualise
├── unet_2CH_TR_best.pth         # Trained weights — 2-Chamber view
├── unet_4CH_TR_best.pth         # Trained weights — 4-Chamber view
├── unet_SAX_TR_best.pth         # Trained weights — Short Axis view
├── training_curves.png          # 3×4 grid of per-epoch loss, Dice, HD95, ASD curves
├── dice_summary.png             # Bar chart — best vs. final val Dice across all views
└── README.md
```

---

## Dataset

Data is sourced from the **CineMulti** dataset on Kaggle (`youssefsharabas/cinemultii`) and consists of NIfTI (`.nii.gz`) Cine MRI images paired with segmentation annotations. Ground-truth LVEF values for Pearson correlation are loaded from `dataset.xlsx`.

Expected directory layout:

```
<root_dir>/
├── SAX_TR/
│   ├── image/    # NIfTI image files
│   └── anno/     # NIfTI annotation files
├── 2CH_TR/
│   ├── image/
│   └── anno/
├── 4CH_TR/
│   ├── image/
│   └── anno/
└── dataset.xlsx  # LVEF ground-truth (sheet per view)
```

---

## Segmentation Classes

| View | Classes | Labels |
|------|---------|--------|
| **SAX** | 4 | 0 = Background · 1 = LV Myocardium · 2 = LV Cavity · 3 = RV Cavity |
| **2CH** | 3 | 0 = Background · 1 = LV Cavity · 2 = LV Myocardium |
| **4CH** | 6 | 0 = Background · 1 = LV Cavity · 2 = LV Myocardium · 3 = RV Cavity · 4 = RA · 5 = LA |

---

## Model Architecture

A classic **U-Net** encoder–decoder:

- **Encoder** — 4 `DoubleConv` blocks (two 3×3 convolutions + BatchNorm + ReLU) each followed by 2×2 MaxPool. Channels grow 64 → 128 → 256 → 512.
- **Bottleneck** — `DoubleConv` at 1024 channels.
- **Decoder** — 4 upsampling blocks using transposed convolutions with skip connections concatenated from the encoder. Channels shrink 512 → 256 → 128 → 64.
- **Output** — 1×1 convolution to the view-specific number of classes.

Input: single-channel grayscale MRI slice `(1 × 256 × 256)`.

---

## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 15 |
| Batch size | 32 |
| Image size | 256 × 256 |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight decay 1e-5) |
| LR scheduler | Cosine Annealing (η_min = 1e-6) |
| Train / Val split | 80 / 20 |

### Loss

Combined **Dice + Cross-Entropy** with equal weighting:

```
L = 0.5 × CrossEntropy + 0.5 × (1 − mean Dice over foreground classes)
```

### Data Augmentation

Applied during training only: random horizontal/vertical flips, random 90° rotations, random brightness/contrast jitter.

### Training Order

Views are trained sequentially: `2CH_TR` → `4CH_TR` → `SAX_TR`.

### Metric Logging

Each view writes a CSV to `/kaggle/working/` as it trains (one row per epoch, flushed immediately so it can be inspected mid-run):

```
metrics_2CH_TR.csv
metrics_4CH_TR.csv
metrics_SAX_TR.csv
```

Columns: `epoch, train_loss, val_loss, val_dice, val_hd95, val_asd, ef_pcc`

The best-Dice checkpoint is saved after each epoch; a final save always occurs at the end of training.

---

## Pre-trained Weights

Three `.pth` files are included, one per view, containing the `state_dict` of the `UNet` model (without the `DataParallel` wrapper):

| File | View | Output classes |
|------|------|---------------|
| `unet_2CH_TR_best.pth` | 2-Chamber | 3 |
| `unet_4CH_TR_best.pth` | 4-Chamber | 6 |
| `unet_SAX_TR_best.pth` | Short Axis | 4 |

Loading example:

```python
model = UNet(in_channels=1, out_channels=4)   # adjust out_channels per view
model.load_state_dict(torch.load("unet_SAX_TR_best.pth", map_location="cpu"))
model.eval()
```

Kaggle model paths used in the notebook:

```
/kaggle/input/models/youssefsharabas/unet/pytorch/default/1/unet_SAX_TR_best.pth
/kaggle/input/models/youssefsharabas/unet/pytorch/default/1/unet_2CH_TR_best.pth
/kaggle/input/models/youssefsharabas/unet/pytorch/default/1/unet_4CH_TR_best.pth
```

---

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **Dice Score** | Overlap between prediction and ground truth, averaged over foreground classes | ↑ higher is better |
| **HD95** | 95th-percentile Hausdorff Distance in pixels — boundary agreement | ↓ lower is better |
| **ASD** | Average Surface Distance — symmetric boundary error | ↓ lower is better |
| **EF PCC** | Pearson correlation between predicted LV cavity pixel count and ground-truth LVEF (SAX only) | ↑ higher is better |

Post-processing: only the largest connected component is retained per class before computing HD95 and ASD.

---

## Output Charts

Two images are produced and saved to `/kaggle/working/`.

### `training_curves.png`
A 3 × 4 grid of line charts, one row per view (SAX / 2CH / 4CH). Each row contains:
- **Train Loss vs. Val Loss** across 15 epochs
- **Val Dice Score** across 15 epochs
- **Val HD95** across 15 epochs
- **Val ASD** — the SAX row also overlays EF PCC on a secondary y-axis

### `dice_summary.png`
A grouped bar chart comparing **Best Val Dice** and **Final Val Dice** side-by-side for all three views, with values labelled on each bar.

---

## Notebook Walkthrough

`Unet_Cine_multi.ipynb` is fully self-contained and runs top-to-bottom:

| Cells | Purpose |
|-------|---------|
| 1–2 | Install PyTorch 2.3.1 + CUDA 11.8, verify GPU |
| 3 | Imports, `VIEW_CONFIG`, set `OUTPUT_DIR` |
| 4 | `UNet` and `DoubleConv` model definition |
| 5 | `DiceCELoss` — combined Dice + CrossEntropy loss |
| 6 | `CineMRIDataset` — NIfTI loading, z-score normalisation, augmentation |
| 7 | Metric helpers — `dice_score`, `hausdorff95_asd`, `keep_largest_component`, `estimate_lv_pixels` |
| 8 | `train_view()` — training loop with per-epoch CSV logging and best-model saving |
| 9 | **Run training** for all three views |
| 10 | Plot and save `training_curves.png` |
| 11 | Plot and save `dice_summary.png` |
| 12 | `evaluate_model()` and `visualize_prediction()` helpers |
| 13 | Set `ROOT` and model paths |
| 14 | Run `evaluate_model()` for all three views |
| 15–17 | Load each model, evaluate, and visualise a sample prediction (MRI / Ground Truth / Prediction) |

---

## Requirements

```
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1
# CUDA 11.8 build:
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
#     --index-url https://download.pytorch.org/whl/cu118

nibabel
opencv-python
pandas
openpyxl
scipy
tqdm
matplotlib
```

---

## Notes

- Images are z-score normalised per slice before being fed to the model.
- SAX volumes are 4D (`H × W × T × S`); the dataset iterates over all time frames and slices.
- 2CH and 4CH volumes are 3D (`H × W × T`); the dataset iterates over all time frames.
- EF PCC is only computed for the SAX view and is logged as `NaN` for 2CH and 4CH.
- The notebook automatically wraps the model with `nn.DataParallel` when multiple GPUs are available.
