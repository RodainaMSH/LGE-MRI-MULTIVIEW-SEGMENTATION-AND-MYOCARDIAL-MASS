# U-Net Cardiac Cine MRI Segmentation — Multi-View

A PyTorch implementation of U-Net for automated cardiac segmentation on multi-view Cine MRI data. Three separate models are trained — one per cardiac view — covering Short Axis (SAX), 2-Chamber (2CH), and 4-Chamber (4CH) orientations.

---

## Project Structure

```
├── Unet_Train_Cine_Multi.ipynb   # Training notebook (all 3 views)
├── Unet_Test_Cine_Multi.ipynb    # Evaluation & visualization notebook
├── unet_2CH_TR_best.pth          # Trained weights — 2-Chamber view
├── unet_4CH_TR_best.pth          # Trained weights — 4-Chamber view
├── unet_SAX_TR_best.pth          # Trained weights — Short Axis view
└── README.md
```

---

## Dataset

The data is sourced from the **CineMulti** dataset on Kaggle (`youssefsharabas/cinemultii`) and consists of NIfTI (`.nii.gz`) Cine MRI images paired with segmentation annotations. Ground-truth Left Ventricular Ejection Fraction (LVEF) values for the SAX view are loaded from `dataset.xlsx`.

Expected directory layout on disk:

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
└── dataset.xlsx  # LVEF ground-truth (SAX sheet used for EF PCC)
```

---

## Segmentation Classes

Each view predicts a different set of cardiac structures:

| View | Classes | Labels |
|------|---------|--------|
| **SAX** | 4 | 0 = Background, 1 = LV Myocardium, 2 = LV Cavity, 3 = RV Cavity |
| **2CH** | 3 | 0 = Background, 1 = LV Cavity, 2 = LV Myocardium |
| **4CH** | 6 | 0 = Background, 1 = LV Cavity, 2 = LV Myocardium, 3 = RV Cavity, 4 = RA, 5 = LA |

---

## Model Architecture

A classic **U-Net** encoder–decoder:

- **Encoder**: 4 downsampling blocks of `DoubleConv` (two 3×3 convolutions + BatchNorm + ReLU) followed by 2×2 MaxPool. Feature channels grow from 64 → 128 → 256 → 512.
- **Bottleneck**: DoubleConv at 1024 channels.
- **Decoder**: 4 upsampling blocks using transposed convolutions, with skip connections concatenated from the encoder at each resolution. Feature channels shrink 512 → 256 → 128 → 64.
- **Output**: 1×1 convolution projecting to the view-specific number of classes.

Input: single-channel grayscale MRI slice (`1 × 256 × 256`).

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

### Loss Function

A combined **Dice + Cross-Entropy** loss with equal weighting (0.5 each):

```
L = 0.5 × CrossEntropy + 0.5 × (1 − mean Dice over foreground classes)
```

### Data Augmentation

Applied during training only: random horizontal/vertical flips, random 90° rotations, and random brightness/contrast jitter.

### Multi-GPU

The training script automatically wraps the model with `nn.DataParallel` when multiple GPUs are detected.

### Saving

The model with the best validation Dice score is saved at the end of each epoch. Final weights are always written at the end of training regardless of best performance.

---

## Pre-trained Weights

Three `.pth` files are included, one per view. They contain the `state_dict` of the `UNet` model (without the `DataParallel` wrapper):

| File | View | Output classes |
|------|------|---------------|
| `unet_2CH_TR_best.pth` | 2-Chamber | 3 |
| `unet_4CH_TR_best.pth` | 4-Chamber | 6 |
| `unet_SAX_TR_best.pth` | Short Axis | 4 |

Loading example:

```python
model = UNet(in_channels=1, out_channels=4)  # adjust out_channels per view
model.load_state_dict(torch.load("unet_SAX_TR_best.pth", map_location="cpu"))
model.eval()
```

---

## Evaluation Metrics

Computed on the validation split during training and in full during testing:

- **Dice Score** — overlap between predicted and ground-truth masks, averaged over foreground classes (higher is better ↑).
- **HD95** — 95th-percentile Hausdorff Distance in pixels, measuring boundary agreement (lower is better ↓).
- **ASD** — Average Surface Distance, a symmetric boundary error metric (lower is better ↓).
- **EF PCC** — Pearson Correlation Coefficient between predicted LV cavity pixel counts and ground-truth LVEF values. Computed for the SAX view only; reported as N/A for 2CH and 4CH.

Post-processing: only the largest connected component is kept per class before computing HD95 and ASD.

---

## Running the Notebooks

### Training (`Unet_Train_Cine_Multi.ipynb`)

1. Set `ROOT` to the path of your dataset root directory.
2. Run all cells. The notebook trains all three views sequentially and saves a `.pth` file for each.

```python
ROOT = "/path/to/cinemultii"
for view in ["SAX_TR", "2CH_TR", "4CH_TR"]:
    train_view(ROOT, view=view, num_epochs=15, batch_size=32)
```

### Testing (`Unet_Test_Cine_Multi.ipynb`)

1. Set `ROOT` to the dataset path and the three model paths (`SAX_MODEL`, `CH2_MODEL`, `CH4_MODEL`).
2. Run `evaluate_model(...)` for quantitative metrics.
3. Run `visualize_prediction(...)` to display side-by-side MRI / ground truth / prediction plots.
4. Optional analysis: `plot_dice_per_sample`, `plot_area_comparison`, `plot_dice_histogram`.

---

## Requirements

```
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1
# CUDA 11.8 build recommended:
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
- SAX volumes can be 4D (`H × W × T × S`); the dataset iterates over all time frames and slices.
- 2CH and 4CH volumes are typically 3D (`H × W × T`); the dataset iterates over all time frames.
- LVEF ground truth is only available for SAX in `dataset.xlsx`; EF PCC is therefore not computed for 2CH or 4CH.
