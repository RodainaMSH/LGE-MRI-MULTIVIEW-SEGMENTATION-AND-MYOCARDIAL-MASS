"""Export per-view segmentation masks in NATIVE geometry as .nii.gz — shared by
the U-Net tracks (nnU-Net already writes native NIfTI itself).

Why this exists: the official Task 2 DSC (70% of the score) is computed on
per-patient NIfTI label masks. The U-Net tracks previously produced only mass
JSONs. This turns their per-slice predictions back into native-resolution label
volumes the challenge (and our unified_eval/compute_dsc.py) can score.

Preprocessing here MUST mirror unet_vanilla/data/dataset_mv.py / the SMP wrapper exactly:
    per-slice min-max to [0,1]  ->  resize to 256 (bilinear)  ->  [tile to 3ch +
    ImageNet-normalize for EfficientNet]  ->  argmax  ->  resize back to native
    (nearest).

"""

from __future__ import annotations

from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def predict_native_volume(model, img_nii, device, tile3: bool = False,
                          size: int = 256) -> np.ndarray:
    """Run `model` slice-by-slice and return a native-geometry (H, W, S) uint8
    label volume. `tile3=True` applies the EfficientNet 3-channel + ImageNet norm.
    """
    data = np.asarray(img_nii.dataobj).astype(np.float32)  # (H, W, S) native
    H, W, S = data.shape
    out = np.zeros((H, W, S), dtype=np.uint8)
    for s in range(S):
        sl = data[:, :, s]
        mn, mx = float(sl.min()), float(sl.max())
        if mx > mn:
            sl = (sl - mn) / (mx - mn)
        inp = cv2.resize(sl, (size, size), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0)  # (1,1,sz,sz)
        if tile3:
            t = t.repeat(1, 3, 1, 1)
            t = (t - IMAGENET_MEAN) / IMAGENET_STD
        t = t.to(device)
        pred = model(t).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)  # (sz,sz)
        out[:, :, s] = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
    return out


def save_mask(volume: np.ndarray, ref_nii, out_path: Path) -> None:
    """Save a label volume with the reference image's affine (same grid as GT)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.uint8), ref_nii.affine), str(out_path))
