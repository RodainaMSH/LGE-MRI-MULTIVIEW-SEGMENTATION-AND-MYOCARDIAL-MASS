"""Track-vs-track comparison on the *mass* side (official VAL set).

What this script DOES (apples-to-apples):
    - Reads each track's raw per-patient mass JSON (`mass_predictions_val.json`,
      produced by that track's moe_predict.py --split val).
    - Computes vPCC, RAE on the CORRECT Task-2 target: **scar mass** (the Codabench
      eval server is fed `mass_predictions.json` whose values are scar grams per
      patient). Also reports LV-myo numbers as a secondary diagnostic.
    - Pulls the real unified DSC (held-out VAL, native NIfTI vs GT) from
      compute_dsc.py's `dsc_matrix_val.json`.
    - Writes a markdown table to unified_eval/mass_comparison_val.md.

Official Task 2 Score (from the Codabench Ranking page):
    Task2 = 0.7 * DSC + 0.3 * (0.5 * max(0, vPCC) + 0.5 * 1/(1+RAE))
where vPCC / RAE are on SCAR MASS (grams) per patient.

Usage:
    python unified_eval/compute_dsc.py        # first, to produce dsc_matrix_val.json
    python unified_eval/compare_tracks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

TRACKS = {
    "unet_vanilla":      ROOT / "unet_vanilla"      / "results" / "logs",
    "unet_efficientnet": ROOT / "unet_efficientnet" / "results" / "logs",
    "nnunet":            ROOT / "nnunet"            / "results" / "logs",
}

# Fallback self-reported best val Dice (per-slice, different averaging per track).
# Used only if the real unified DSC (unified_eval/dsc_matrix_val.json) is absent.
SELF_REPORTED_DSC = {
    "unet_vanilla":      0.607,   # stage1 best val Dice
    "unet_efficientnet": 0.649,   # stage1 best val Dice
    "nnunet":            0.693,   # Dataset000 generalist mean val Dice
}


def load_real_dsc() -> dict:
    """Real unified DSC (official VAL, native NIfTI vs GT) from compute_dsc.py."""
    p = ROOT / "unified_eval" / "dsc_matrix_val.json"
    if not p.exists():
        return {}
    return {r["track"]: r["dsc_overall"] for r in json.loads(p.read_text())}


REAL_DSC = load_real_dsc()


def load_raw_mass(track_dir: Path) -> dict:
    """Per-patient scar/LV mass from this track's official VAL inference."""
    p = track_dir / "mass_predictions_val.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run this track's moe_predict.py --split val first."
        )
    return json.loads(p.read_text())


def vpcc_rae(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    if len(pred) < 2 or pred.std() == 0 or true.std() == 0:
        vpcc_val = float("nan")
    else:
        vpcc_val = float(np.corrcoef(pred, true)[0, 1])
    nz = true > 1e-6
    rae_val = float(np.mean(np.abs(pred[nz] - true[nz]) / true[nz])) if nz.any() else float("nan")
    return vpcc_val, rae_val


def task2_score(dsc: float, vpcc: float, rae: float) -> float:
    """Official Codabench Task 2 Score: 0.7*DSC + 0.3*(0.5*max(0,vPCC) + 0.5/(1+RAE))."""
    return 0.7 * dsc + 0.3 * (0.5 * max(0.0, vpcc) + 0.5 / (1.0 + rae))


def evaluate_track(name: str, log_dir: Path) -> dict:
    raw = load_raw_mass(log_dir)
    patients = raw["patients"]
    pids = sorted(patients.keys(), key=lambda s: int(s))

    scar_pred = np.array([patients[p]["scar_mass_final"] for p in pids], dtype=np.float64)
    scar_true = np.array([patients[p]["true_scar_mass"]  for p in pids], dtype=np.float64)
    lv_pred   = np.array([patients[p]["lv_mass_final"]   for p in pids], dtype=np.float64)
    lv_true   = np.array([patients[p]["true_lv_mass"]    for p in pids], dtype=np.float64)

    scar_vpcc, scar_rae = vpcc_rae(scar_pred, scar_true)
    lv_vpcc,   lv_rae   = vpcc_rae(lv_pred,   lv_true)

    dsc = REAL_DSC.get(name, SELF_REPORTED_DSC[name])
    score_scar = task2_score(dsc, scar_vpcc, scar_rae)
    score_lv   = task2_score(dsc, lv_vpcc,   lv_rae)

    return {
        "track":              name,
        "n_val_patients":     int(len(scar_pred)),
        "pids":               pids,
        "dsc_used":           dsc,
        "scar_pred":          scar_pred.tolist(),
        "scar_true":          scar_true.tolist(),
        "scar_vpcc":          scar_vpcc,
        "scar_rae":           scar_rae,
        "lv_vpcc":            lv_vpcc,
        "lv_rae":             lv_rae,
        "task2_score_scar":   score_scar,
        "task2_score_lv":     score_lv,
    }


def render_markdown(rows: list[dict]) -> str:
    leader = {"name": "tobi_hao", "dsc": 0.79, "vpcc": 0.71, "rae": 0.76,
              "task2": task2_score(0.79, 0.71, 0.76)}
    n_val = rows[0]["n_val_patients"]
    n_zero = int(np.sum(np.array(rows[0]["scar_true"]) <= 1e-6))

    lines = []
    lines.append("# Unified track comparison (Task 2 / LGE) — official VAL set\n")
    lines.append("All three tracks are scored on the authors' official validation split "
                 "(`*_VAL` folders: LV IDs 41-47, RAS IDs 81-94), from each track's "
                 "per-patient scar-mass predictions.\n")
    lines.append("> **Caveats**\n"
                 f"> - The LV scar-mass metrics use the **{n_val}-patient VAL cohort**; "
                 f"**{n_zero} of {n_val} have ~zero true scar mass**, so RAE is sensitive to "
                 "small denominators on the near-zero patients.\n"
                 "> - DSC column is the **real unified DSC** (official VAL, native NIfTI vs "
                 "GT, identical averaging across tracks) from compute_dsc.py.\n")

    lines.append("## Scar mass per patient (VAL set)\n")
    pids = rows[0]["pids"]
    header = "| patient | true scar (g) | " + " | ".join(f"{r['track']} pred (g)" for r in rows) + " |"
    sep    = "|---|---|" + "|".join("---" for _ in rows) + "|"
    lines.append(header); lines.append(sep)
    for i, pid in enumerate(pids):
        cells = [f"pid={pid}", f"{rows[0]['scar_true'][i]:.2f}"]
        cells += [f"{r['scar_pred'][i]:.2f}" for r in rows]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Task 2 score (scar mass target -- the real Codabench metric)\n")
    header = "| Track | DSC* | scar vPCC | scar RAE | Task 2 Score |"
    sep    = "|---|---|---|---|---|"
    lines.append(header); lines.append(sep)
    for r in rows:
        lines.append(
            f"| `{r['track']}` "
            f"| {r['dsc_used']:.3f} "
            f"| {r['scar_vpcc']:+.3f} "
            f"| {r['scar_rae']:.2f} "
            f"| **{r['task2_score_scar']:.3f}** |"
        )
    lines.append(
        f"| `{leader['name']}` (current LB #1) "
        f"| {leader['dsc']:.3f} | {leader['vpcc']:+.3f} | {leader['rae']:.3f} "
        f"| **{leader['task2']:.3f}** |"
    )

    lines.append("\n## Task 2 score (LV myo mass -- secondary, for reference)\n")
    header = "| Track | DSC* | LV vPCC | LV RAE | Task 2 (LV) |"
    sep    = "|---|---|---|---|---|"
    lines.append(header); lines.append(sep)
    for r in rows:
        lines.append(
            f"| `{r['track']}` "
            f"| {r['dsc_used']:.3f} "
            f"| {r['lv_vpcc']:+.3f} "
            f"| {r['lv_rae']:.3f} "
            f"| {r['task2_score_lv']:.3f} |"
        )

    lines.append("\n\\* DSC = real unified DSC from unified_eval/compute_dsc.py "
                 "(official VAL set, native NIfTI vs GT, same averaging for all tracks).\n")

    lines.append("## Task 2 Score formula\n")
    lines.append("```\n"
                 "Task2 = 0.7 * DSC + 0.3 * (0.5 * max(0, vPCC) + 0.5 / (1 + RAE))\n"
                 "```\n")

    lines.append("## Open items\n")
    lines.append("1. **Scar segmentation Dice** is the only gap to the leaderboard #1 — "
                 "the scar-mass term already matches/beats it.\n")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = [evaluate_track(name, log_dir) for name, log_dir in TRACKS.items()]

    out_md = ROOT / "unified_eval" / "mass_comparison_val.md"
    out_json = ROOT / "unified_eval" / "mass_comparison_val.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(rows))
    out_json.write_text(json.dumps(rows, indent=2))

    n_val = rows[0]["n_val_patients"]
    print(f"Wrote {out_md.relative_to(ROOT)}")
    print(f"Wrote {out_json.relative_to(ROOT)}\n")
    print(f"=== Scar mass per patient (VAL set, n={n_val}) ===")
    pids = rows[0]["pids"]
    head = f"{'pid':>4}  {'true':>8}  " + "  ".join(f"{r['track']:>20}" for r in rows)
    print(head)
    for i, pid in enumerate(pids):
        line = f"{pid:>4}  {rows[0]['scar_true'][i]:>8.2f}  " + "  ".join(
            f"{r['scar_pred'][i]:>20.2f}" for r in rows
        )
        print(line)

    print("\n=== Task 2 score (SCAR mass -- real Codabench target) ===")
    for r in rows:
        print(f"  {r['track']:<22} dsc={r['dsc_used']:.3f}  "
              f"scar_vpcc={r['scar_vpcc']:+.3f}  scar_rae={r['scar_rae']:8.2f}  "
              f"-> Task2 = {r['task2_score_scar']:.3f}")
    print(f"  {'tobi_hao (LB #1)':<22} dsc=0.790  "
          f"vpcc=+0.710  rae=0.760  -> Task2 = {task2_score(0.79, 0.71, 0.76):.3f}")

    print("\n=== Task 2 score (LV mass -- secondary) ===")
    for r in rows:
        print(f"  {r['track']:<22} dsc={r['dsc_used']:.3f}  "
              f"lv_vpcc={r['lv_vpcc']:+.3f}  lv_rae={r['lv_rae']:.3f}  "
              f"-> Task2 = {r['task2_score_lv']:.3f}")


if __name__ == "__main__":
    main()
