"""Anatomically-constrained scar-label deformation — the PLACEMENT fix (replaces v1's random transplant).

Re-implements the proven EMIDEC recipe (Lustermans et al., CMPB 2022, Sec 2.3 / Suppl. Table 2), whose
exact code was NOT shipped in CMRISynthSeg (only the SPADE renderer was). Verbatim from the paper:
  "The augmentation of the scar segmentations included rotation by a multiple of 60 deg, elastic
   deformation, dilation and opening ... to create previously unseen shapes and positioning of scar."

Why this fixes v1: v1 moved a scar shape to a RANDOM myocardial-wall pixel (anatomically impossible).
Here we DEFORM the patient's OWN scar IN PLACE:
  - rotation is about the LV-cavity centroid -> the scar stays on the SAME myocardial ring, only its
    ANGULAR sector changes (k*60 deg ~ moving between AHA segments). Radius from centre is preserved,
    so a subendocardial scar stays subendocardial, a transmural one stays transmural.
  - elastic + dilation/opening reshape it (new shape / extent) while staying contiguous.
  - the result is INTERSECTED with the wall (myo u scar) so scar can never leave the ring; the original
    scar voxels revert to myocardium. Mutually exclusive with myo, on-ring, contiguous = realistic.

Class scheme (ours == theirs): 0 bg, 1 LVcav, 2 LVmyo, 3 scar, 4 RVcav.
"""
import numpy as np
from scipy import ndimage


def _centroid(mask):
    ys, xs = np.where(mask)
    return np.array([ys.mean(), xs.mean()]) if len(ys) else None


def _rotate_about(mask, center, deg, order=0):
    """Rotate a 2D array about an arbitrary point `center` (y,x). Binary -> order=0 (nearest)."""
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    # affine_transform maps output coord o -> input coord R@o + offset; fix center: c = R@c + offset
    offset = center - R @ center
    return ndimage.affine_transform(mask.astype(np.float32), R, offset=offset, order=order,
                                    mode="constant", cval=0.0)


def _elastic(mask, rng, alpha, sigma):
    """Mild elastic deformation of a binary mask (random smoothed displacement field)."""
    h, w = mask.shape
    dy = ndimage.gaussian_filter((rng.random((h, w)) * 2 - 1), sigma) * alpha
    dx = ndimage.gaussian_filter((rng.random((h, w)) * 2 - 1), sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack([yy + dy, xx + dx])
    return ndimage.map_coordinates(mask.astype(np.float32), coords, order=0, mode="constant", cval=0.0)


def deform_scar(lab, rng, p_elastic=0.6, p_morph=0.6):
    """Return a NEW label with the scar deformed in place (on-ring), or None if not applicable.

    `lab` : 2D int array in {0,1,2,3,4}.  `rng` : np.random.Generator.
    Guarantees: scar subset of wall (myo u original-scar); original scar voxels -> myo; >= ~half the
    original scar mass survives (else None, so we never emit a near-empty/garbage label).
    """
    scar = (lab == 3)
    if scar.sum() < 20:
        return None
    wall = (lab == 2) | (lab == 3)                 # the myocardial ring (myo + existing scar)
    if wall.sum() < 50:
        return None

    # rotation centre = LV cavity centroid (the ring centre); fall back to wall centroid
    center = _centroid(lab == 1)
    if center is None:
        center = _centroid(wall)

    # --- 1. rotation by a multiple of 60 deg about the ring centre (k in 1..5 -> always a new sector) ---
    k = int(rng.integers(1, 6))
    m = _rotate_about(scar.astype(np.float32), center, 60.0 * k, order=0) > 0.5

    # --- 2. elastic deformation (optional) ---
    if rng.random() < p_elastic:
        m = _elastic(m, rng, alpha=rng.uniform(3.0, 8.0), sigma=rng.uniform(4.0, 7.0)) > 0.5

    # --- 3. dilation OR opening (optional) — changes extent/shape, paper uses both ---
    if rng.random() < p_morph:
        r = int(rng.integers(1, 3))
        st = ndimage.generate_binary_structure(2, 1)
        st = ndimage.iterate_structure(st, r)
        m = ndimage.binary_dilation(m, st) if rng.random() < 0.5 else ndimage.binary_opening(m, st)

    # --- 4. constrain to the ring; original scar reverts to myo ---
    newscar = m & wall
    if newscar.sum() < max(20, 0.5 * scar.sum()):  # reject degenerate placements (kept too little)
        return None

    out = lab.copy()
    out[lab == 3] = 2                              # erase original scar -> myo
    out[newscar] = 3                               # paint deformed scar on the ring
    return out
