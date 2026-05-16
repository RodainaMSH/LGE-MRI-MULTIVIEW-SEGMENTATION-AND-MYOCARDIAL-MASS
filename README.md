# LGE MRI Multi-View Segmentation and Myocardial Mass Quantification

**Authors:** Youssef Araby, Rodaina Hebishy,Youssef hazem, Mihael atef
**Institution:** Nile University, Department of Computer Science  
**Course:** Deep Learning  
**Date:** May 2026

---

## Abstract

This project presents automated segmentation and quantification methods for myocardial scar tissue from multi-view Late Gadolinium Enhancement (LGE) cardiac MRI. Two deep learning architectures are implemented and evaluated: a standard U-Net with patient-specific voxel correction (Hebishy) and an Attention U-Net with k-fold cross-validation (Araby). Both models are trained and validated on the CMR-MULTI Challenge dataset, demonstrating strong performance in clinical mass quantification tasks.

## Project Overview

This research implements deep learning-based segmentation of cardiac structures and myocardial scar tissue from multi-view LGE MRI scans, with emphasis on accurate volumetric mass quantification for clinical deployment.

### Task
- **Input**: Multi-view LGE MRI scans (SAX, 2CH, 4CH, RAS views)
- **Output**: 5-class segmentation
  - 0: Background
  - 1: Left Ventricle Cavity
  - 2: Left Ventricle Myocardium
  - 3: Myocardial Scar (primary target)
  - 4: Right Ventricle Cavity

### Dataset
- **Source**: CMR-MULTI Challenge - LGE_MULTI
- **Total Slices**: 1,220 2D slices from 4 cardiac views
- **Split**: 976 training / 244 validation (80/20)
- **Views**:
  - SAX (Short Axis): 400 slices
  - 2CH (2-Chamber): 120 slices
  - 4CH (4-Chamber): 120 slices
  - RAS (Radial): 580 slices

## Experimental Results

### Run 6: U-Net with Patient-Specific Voxel Correction (Hebishy)

| Metric | Value | Clinical Target | Status |
|--------|-------|-----------------|--------|
| **Dice Score** | 0.7560 | > 0.75 | ✅ Achieved |
| **vPCC** | 0.9077 | > 0.90 | ✅ Exceeded |
| **RAE** | 0.3735 | < 0.40 | ✅ Achieved |

**Key Innovation:** Patient-specific voxel volume extraction from DICOM metadata, achieving 22.5% improvement in RAE compared to hardcoded spacing assumptions.

**Architecture:**
- Standard U-Net with 5-level encoder-decoder
- 31.0M trainable parameters
- Focal Loss (γ=2.0, α=[1,2,2,10,2])
- Training: 50 epochs, 92 minutes on Apple M4 Pro

### Run 4: Attention U-Net with K-Fold Cross-Validation (Araby)

| Metric | Value | Clinical Target | Status |
|--------|-------|-----------------|--------|
| **Dice Score** | 0.8292 ± 0.0331 | > 0.75 | ✅ Excellent |
| **vPCC** | 0.9047 ± 0.0055 | > 0.90 | ✅ Achieved |
| **RAE** | 0.4153 ± 0.0379 | < 0.40 | Close (3.8% above) |

**Key Innovation:** Attention gates for improved boundary refinement with robust 5-fold cross-validation.

**Architecture:**
- Attention U-Net with spatial attention mechanisms
- 34.2M trainable parameters
- Focal Loss (γ=2.0, α=[1,2,2,10,2])
- Training: 5-fold CV, 768 minutes total

## Repository Structure

```
├── attention_unet/                   # Araby's Attention U-Net experiment
│   ├── README_RUN4.md               # Detailed documentation
│   ├── RESULTS_RUN4.txt             # Complete results analysis
│   ├── TRAINING_LOG_RUN4.txt        # Epoch-by-epoch training log
│   ├── attention_unet.py            # Attention U-Net architecture
│   ├── train_attention_unet_kfold.py # 5-fold CV training script
│   ├── fold1_history.json           # Fold 1 training metrics
│   ├── fold2_history.json           # Fold 2 training metrics
│   ├── fold3_history.json           # Fold 3 training metrics
│   └── fold5_history.json           # Fold 5 training metrics
│
├── unet_voxel_correction/           # Hebishy's U-Net experiment
│   ├── README_RUN6.md               # Detailed documentation
│   ├── RESULTS_RUN6.txt             # Complete results analysis
│   ├── model.py                     # Standard U-Net architecture
│   ├── train_unet_corrected.py      # Training script with voxel correction
│   └── training_history_run6.json   # Training metrics
│
└── dataset.py                        # Multi-view LGE-MRI data loader
```

## Usage

### Training U-Net with Voxel Correction (Run 6)
```bash
python unet_voxel_correction/train_unet_corrected.py
```

### Training Attention U-Net with K-Fold CV (Run 4)
```bash
python attention_unet/train_attention_unet_kfold.py
```

### Inference Example
```python
import torch
from unet_voxel_correction.model import UNet

# Load best clinical model
model = UNet(in_channels=1, num_classes=5)
checkpoint = torch.load('unet_voxel_correction/best_model_run6.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Perform inference
with torch.no_grad():
    output = model(image_tensor)
    segmentation = output.argmax(dim=1)
```

## Methodology

### Data Preprocessing
- Input normalization: min-max scaling to [0, 1]
- Resizing to 256×256 pixels
- Multi-view data integration (SAX, 2CH, 4CH, RAS)

### Data Augmentation
- Random horizontal/vertical flips (p=0.5)
- Random rotation (±15°)
- Brightness/contrast adjustment
- Gaussian noise and blur
- Elastic deformations

### Loss Function
**Focal Loss** with γ=2.0 to address severe class imbalance:
- Class weights: [1.0, 2.0, 2.0, 10.0, 2.0]
- 10× emphasis on scar tissue (< 6% of pixels)

### Training Configuration
- Optimizer: AdamW (LR=1e-4, weight decay=1e-4)
- Learning rate warm-up: 5 epochs
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Batch size: 8
- Hardware: Apple M4 Pro, 64GB unified memory

### Evaluation Metrics
- **Dice Score**: Segmentation overlap coefficient
- **vPCC**: Volumetric Pearson correlation for mass quantification
- **RAE**: Relative absolute error in mass estimation

## Key Findings

### Clinical Mass Quantification
**Run 6 (Hebishy)** is the only experiment achieving both clinical targets:
- ✅ vPCC > 0.9 (achieved 0.9077)
- ✅ RAE < 0.4 (achieved 0.3735)

### Voxel Correction Impact
Patient-specific voxel calibration provides **22.5% improvement** in RAE:
- Baseline (hardcoded 1.5×1.5×10mm): RAE = 0.4820
- Patient-specific (DICOM metadata): RAE = 0.3735

**Insight:** Spatial calibration is more critical than architectural complexity for volumetric accuracy.

### Segmentation vs. Mass Accuracy
**Run 4 (Araby)** achieves superior segmentation but lower mass accuracy:
- Higher Dice: 0.8292 vs 0.7560
- Lower mass accuracy: RAE 0.4153 vs 0.3735

**Conclusion:** Pixel-level segmentation quality does not guarantee volumetric quantification accuracy.

## Clinical Applications

**Diagnostic Capabilities:**
- ✅ Myocardial scar detection and localization
- ✅ Patient risk stratification by scar burden
- ✅ Volumetric scar mass quantification (vPCC 0.9077)
- ✅ Multi-view cardiac assessment

**Clinical Validation:**
- Both correlation (vPCC > 0.9) and error (RAE < 0.4) targets met in Run 6
- Suitable for clinical deployment with appropriate validation protocols

## Requirements

**Python Environment:**
- Python 3.11+
- PyTorch 2.0+
- torchvision
- numpy
- nibabel (DICOM/NIfTI processing)
- albumentations (data augmentation)
- scikit-learn
- scipy
- matplotlib
- tqdm

**Hardware:**
- Recommended: Apple M4 Pro (64GB RAM) or NVIDIA GPU (16GB+ VRAM)
- Minimum: 16GB RAM, GPU with 8GB VRAM

## Installation

```bash
git clone https://github.com/RodainaMSH/LGE-MRI-MULTIVIEW-SEGMENTATION-AND-MYOCARDIAL-MASS
cd LGE-MRI-MULTIVIEW-SEGMENTATION-AND-MYOCARDIAL-MASS
pip install -r requirements.txt
```

## Dataset

**CMR-MULTI Challenge - LGE_MULTI**
- 1,220 LGE-MRI slices from multi-view cardiac imaging
- 976 training / 244 validation (80/20 split)
- 4 views: SAX (400), 2CH (120), 4CH (120), RAS (580)

## Authors & Contributions

**Rodaina Hebishy**
- U-Net architecture implementation
- Patient-specific voxel correction from DICOM metadata
- Unet experiment achieving both clinical targets

**Youssef Araby**
- Attention U-Net architecture with attention gates
- 5-fold cross-validation implementation
- AttentionUnet experiment achieving highest segmentation accuracy

**Youssef Hazem**
-U-Net Cardiac Cine MRI Segmentation — Multi-View


## Acknowledgments

This work was completed as part of the Deep Learning course at Nile University, Department of Computer Science. Dataset provided by the CMR-MULTI Challenge.

## License

This project is released for academic and research purposes.

## Contact

For questions or collaboration:
- Rodaina Hebishy: [GitHub: RodainaMSH]
- Youssef Araby: [GitHub: youssef-Araby]
- Youssef Hazem: [GitHub: youssefsharabas]
