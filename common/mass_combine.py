"""Combine per-view scar-mass estimates into ONE number per patient.

Each LGE view (SAX/2CH/4CH) yields its own scar-mass estimate from its own
segmentation; they disagree because each view images the same heart differently
(e.g. for val pid 47, true=138 g, the views say SAX=75 / 2CH=128 / 4CH=93 g).
This module turns those per-view grams into ONE final number for the patient.

This is the rebuilt, renamed successor to the deleted MoE "fusion". IMPORTANT
distinction: there are NO expert *models* here — masks come from the plain
generalist (experts didn't beat it). This is purely a multi-view MASS combiner,
which DID earn its keep: on the 7 val patients, combining the views beat trusting
a single view's scar mass (verified 2026-06-06). Choose the strategy + native view
by 5-fold CV on the 40 TRAIN patients, never on the 7 noisy val patients.

All functions take `masses` = {"SAX": g, "2CH": g, "4CH": g} (any subset present)
and return one float. `native` = the view to anchor on (validated, not assumed).
"""

from __future__ import annotations

from statistics import median
from typing import Mapping

VIEWS = ("SAX", "2CH", "4CH")
_EPS = 1e-6


def _present(masses: Mapping[str, float]) -> dict[str, float]:
    return {v: float(masses[v]) for v in VIEWS if v in masses and masses[v] is not None}


def single_view(masses: Mapping[str, float], view: str) -> float:
    """Trust one view's estimate only (the simplest baseline)."""
    return float(masses[view])


def uniform_mean(masses: Mapping[str, float]) -> float:
    m = _present(masses)
    return sum(m.values()) / len(m) if m else 0.0


def median_combine(masses: Mapping[str, float]) -> float:
    m = _present(masses)
    return float(median(m.values())) if m else 0.0


def native_weighted(masses: Mapping[str, float], native: str,
                    native_weight: float = 0.55,
                    disagreement: bool = True) -> float:
    """Anchor on `native` (weight `native_weight`); split the rest among the others.

    If `disagreement` is on, shrink each non-native view by 1/(1+|m-consensus|/consensus)
    so a view that wildly disagrees with the group consensus (median) is trusted less.
    Weights are renormalized; returns the weighted sum. This is the rebuilt logic of
    the old `fuse_three_experts` (native_weight=0.55 + inverse-deviation shrink).
    """
    m = _present(masses)
    if not m:
        return 0.0
    if native not in m or len(m) == 1:
        return uniform_mean(m)
    consensus = float(median(m.values()))
    others = [v for v in m if v != native]
    base_other = (1.0 - native_weight) / len(others)
    w = {native: native_weight}
    for v in others:
        shrink = 1.0 / (1.0 + abs(m[v] - consensus) / (consensus + _EPS)) if disagreement else 1.0
        w[v] = base_other * shrink
    total = sum(w.values())
    return sum(w[v] / total * m[v] for v in m)


# Registry of strategies the validation harness sweeps. Each maps masses(+native) -> float.
def strategies(native: str = "4CH") -> dict:
    return {
        "SAX_only": lambda m: single_view(m, "SAX"),
        "2CH_only": lambda m: single_view(m, "2CH"),
        "4CH_only": lambda m: single_view(m, "4CH"),
        "uniform_mean": uniform_mean,
        "median": median_combine,
        f"native_weighted({native})": lambda m: native_weighted(m, native),
        f"native_weighted_nodisagree({native})": lambda m: native_weighted(m, native, disagreement=False),
    }
