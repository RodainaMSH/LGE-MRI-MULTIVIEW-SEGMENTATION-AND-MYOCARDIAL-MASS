"""Evaluation metrics — per-view, per-class, per-patient.

All functions are pure (no I/O), so they can be unit-tested and reused from
training and inference.

Metrics provided:
    - dice_per_class(pred, target, num_classes)        per-class Dice on tensors
    - dice_macro(pred, target, num_classes)            macro-Dice (avg over present classes)
    - patient_mass_grams(mask, vox_vol_cm3, classes)   pixel-count -> grams
    - vpcc(pred_arr, true_arr)                          volumetric Pearson correlation
    - rae(pred_arr, true_arr)                           relative absolute error
    - aggregate_patient_masses(preds, gts, voxel_vols, patient_ids, classes)
                                                        slice-level -> patient-level

"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------
def dice_per_class(pred: torch.Tensor, target: torch.Tensor,
                   num_classes: int, eps: float = 1e-6,
                   skip_background: bool = True) -> dict[int, float]:
    """Per-class Dice on integer-class tensors of shape (B, H, W).

    Only classes that appear in either pred or target contribute. Background
    (class 0) is skipped by default.
    """
    out: dict[int, float] = {}
    start = 1 if skip_background else 0
    for c in range(start, num_classes):
        p = (pred == c)
        t = (target == c)
        union = p.sum().float() + t.sum().float()
        if union == 0:
            continue
        inter = (p & t).sum().float()
        out[c] = ((2 * inter + eps) / (union + eps)).item()
    return out


def dice_macro(pred: torch.Tensor, target: torch.Tensor,
               num_classes: int, skip_background: bool = True) -> float:
    """Mean Dice across classes that are present (NaN-safe)."""
    pc = dice_per_class(pred, target, num_classes, skip_background=skip_background)
    return float(np.mean(list(pc.values()))) if pc else float("nan")


# ---------------------------------------------------------------------------
# Mass calculation (pixel-count -> grams using DICOM voxel volume)
# ---------------------------------------------------------------------------
MYOCARDIAL_DENSITY_G_PER_CM3 = 1.05  # standard value used in Phase 1


def patient_mass_grams(mask: np.ndarray, vox_vol_cm3: float,
                       classes: Iterable[int],
                       density: float = MYOCARDIAL_DENSITY_G_PER_CM3) -> float:
    """Mass in grams = (sum of pixels in `classes`) * voxel_volume_cm3 * density."""
    pixels = np.isin(mask, list(classes)).sum()
    return float(pixels * vox_vol_cm3 * density)


def aggregate_patient_masses(
    pred_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    voxel_volumes_cm3: Sequence[float],
    patient_ids: Sequence[int],
    classes: Iterable[int] = (3,),  # default = scar
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Sum per-slice mass into per-patient mass for pred and gt.

    Returns (pred_array, true_array, ordered_patient_ids) of equal length.
    """
    pred: dict[int, float] = defaultdict(float)
    true: dict[int, float] = defaultdict(float)
    classes = list(classes)

    for pm, gm, vv, pid in zip(pred_masks, gt_masks, voxel_volumes_cm3, patient_ids):
        pred[int(pid)] += patient_mass_grams(pm, float(vv), classes)
        true[int(pid)] += patient_mass_grams(gm, float(vv), classes)

    pids = sorted(pred.keys())
    return (
        np.array([pred[p] for p in pids], dtype=np.float64),
        np.array([true[p] for p in pids], dtype=np.float64),
        pids,
    )


# ---------------------------------------------------------------------------
# Clinical mass metrics
# ---------------------------------------------------------------------------
def vpcc(pred: np.ndarray, true: np.ndarray) -> float:
    """Volumetric Pearson correlation between predicted and true patient masses."""
    if len(pred) < 2 or pred.std() == 0 or true.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def rae(pred: np.ndarray, true: np.ndarray, eps: float = 1e-6) -> float:
    """Mean relative absolute error: mean(|pred - true| / max(true, eps)).

    Patients with true == 0 are excluded (relative error is undefined there).
    Returns NaN if no patient has nonzero true mass.
    """
    nonzero = true > eps
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs(pred[nonzero] - true[nonzero]) / true[nonzero]))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    pred = torch.randint(0, 5, (2, 64, 64))
    target = torch.randint(0, 5, (2, 64, 64))
    pc = dice_per_class(pred, target, num_classes=5)
    print(f"per-class Dice (random pred/target): {pc}")
    print(f"macro Dice:                          {dice_macro(pred, target, 5):.4f}")

    rng = np.random.default_rng(0)
    pred_mass = rng.uniform(5, 25, 8)
    true_mass = pred_mass + rng.normal(0, 2, 8)
    print(f"\nvPCC(pred, true) = {vpcc(pred_mass, true_mass):.4f}")
    print(f"RAE (pred, true) = {rae(pred_mass, true_mass):.4f}")
    print("\ncommon/metrics.py smoke test passed.")
