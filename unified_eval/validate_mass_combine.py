"""Validate multi-view scar-mass COMBINING strategies against single-view baselines.

Reads a per-view scar-mass JSON (the format written by nnunet_mass.py /
the old mass_predictions_*.json):
    patients[pid].scar_mass_per_expert = {"SAX":g, "2CH":g, "4CH":g}
    patients[pid].true_scar_mass       = g
For every strategy in common.mass_combine it computes the Task-2 clinical term
(vPCC + RAE on scar grams) and the resulting Task-2 score (DSC held fixed, since
the masks don't change with how you combine the mass numbers).

Use on the 7 VAL patients only as a sanity check (noisy: 5 nonzero-scar patients).
The HONEST decision must be made on the 40 TRAIN patients using OUT-OF-FOLD
predictions (5-fold), so the per-view masses reflect real generalization error.

Usage:
    PYTHONPATH=. python unified_eval/validate_mass_combine.py \
        --mass-json nnunet/results/logs/mass_predictions_val.json --dsc 0.733 --native 4CH
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common.metrics import vpcc, rae
from common.mass_combine import strategies


def clinical_term(v: float, r: float) -> float:
    return 0.5 * max(0.0, v) + 0.5 / (1.0 + r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass-json", required=True)
    ap.add_argument("--dsc", type=float, default=0.733,
                    help="fixed DSC for the Task-2 column (masks don't change with the mass combiner)")
    ap.add_argument("--native", default="4CH", help="anchor view for native_weighted strategies")
    args = ap.parse_args()

    d = json.load(open(args.mass_json))
    P = d["patients"]
    pids = sorted(P, key=int)
    true = np.array([float(P[p]["true_scar_mass"]) for p in pids])
    per_view = [P[p]["scar_mass_per_expert"] for p in pids]
    n_nonzero = int((true > 1e-6).sum())

    print(f"mass-json = {args.mass_json}")
    print(f"{len(pids)} patients ({n_nonzero} with nonzero true scar -> drive RAE); "
          f"DSC fixed at {args.dsc}\n")
    print(f"{'strategy':<32} {'vPCC':>7} {'RAE':>7} {'clin(30%)':>10} {'Task2':>8}")
    print("-" * 70)

    rows = []
    for name, fn in strategies(args.native).items():
        pred = np.array([float(fn(m)) for m in per_view])
        v, r = vpcc(pred, true), rae(pred, true)
        term = clinical_term(v, r)
        task2 = 0.7 * args.dsc + 0.3 * term
        rows.append((name, v, r, term, task2))

    for name, v, r, term, task2 in sorted(rows, key=lambda x: -x[4]):
        print(f"{name:<32} {v:>7.3f} {r:>7.3f} {term:>10.3f} {task2:>8.4f}")

    best = max(rows, key=lambda x: x[4])
    print(f"\nbest here: {best[0]}  (Task2 {best[4]:.4f}).  "
          f"On N={len(pids)} treat as indicative; decide on 40-TRAIN out-of-fold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
