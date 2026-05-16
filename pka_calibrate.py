#!/usr/bin/env python3
"""
pka_calibrate.py  —  Functional-group LFER calibration for pKa prediction
==========================================================================
Fits one linear free energy relationship (LFER) per functional group type:

    pKa_pred = a_fg × pKa_calc + b_fg

where:
    pKa_calc = ΔG_DFT / (RT ln10)   [uncorrected, ~190–215]
    a_fg     = slope     (corrects scale; encodes Born solvation
                          variation across charge states of the same group)
    b_fg     = intercept (corrects average G*(H+) offset)
    fg       = functional group label (e.g. "carboxylate", "ammonium", ...)

Why functional groups rather than charge transitions
----------------------------------------------------
The charge state of the molecule shifts the raw DFT energy scale (Born
solvation scales as q²), but the functional group determines which atoms
are involved in the proton transfer and hence the chemical environment.
A carboxylate deprotonation on a cation (+2→+1) and on a neutral molecule
(0→-1) differ in pKa_calc by ~14 units, but this difference is a
systematic Born effect that the LFER slope absorbs automatically.

Grouping by functional group therefore:
  1. Produces more physically interpretable parameters.
  2. Allows training data from different charge states to contribute
     to the same LFER, increasing the pKa_calc span and improving the
     slope estimate.
  3. Maps cleanly to how chemists think about ionisation.

Standard functional groups
--------------------------
  carboxylate   RCOOH → RCOO⁻ + H⁺            pKa ~1–6
  ammonium      RNH₃⁺ → RNH₂ + H⁺  (aliphatic) pKa ~8–11
  imidazolium   ImH⁺  → Im   + H⁺             pKa ~5–8
  guanidinium   Arg side chain                  pKa ~12–13
  phenol        ArOH  → ArO⁻ + H⁺              pKa ~7–11
  thiol         RSH   → RS⁻  + H⁺              pKa ~8–11
  Custom labels are also accepted.

.msf file format additions
--------------------------
  Training compound base microstates need both pka_step and functional_group:

    [microstate]
    name             = Acetate_Ac-
    charge           = -1
    multiplicity     = 1
    n_protons        = 0
    pka_step         = 4.756          # experimental pKa for this step
    functional_group = carboxylate    # group being deprotonated

  Target compound base microstates need only functional_group:

    [microstate]
    name             = Anion
    charge           = -1
    multiplicity     = 1
    n_protons        = 0
    functional_group = carboxylate    # used to select LFER parameters

Caching
-------
  Each compound's G_eff values are cached as <compound>_geff.json.
  Re-running with additional .msf files reuses cached results for
  all previously computed compounds.

Usage
-----
  # Run calibration:
  python pka_calibrate.py acetic_acid.msf glycine.msf histidine.msf \\
      --orca /apps/software/orca/orca-6.1.0/orca --nprocs 12

  # Add more compounds later (existing results loaded from cache):
  python pka_calibrate.py acetic_acid.msf glycine.msf histidine.msf \\
      formic_acid.msf aspartic_acid.msf \\
      --orca /apps/software/orca/orca-6.1.0/orca --nprocs 12

  # Check which functional groups are calibrated:
  python pka_calibrate.py --check pka_results/lfer_params.json

  # Dry-run (no ORCA):
  python pka_calibrate.py acetic_acid.msf histidine.msf --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from pka_calc import (
    Microstate, parse_msf, run_ensemble, _mock_run,
    KB_EV, EH_TO_EV, EH_TO_KCAL, LN10, TEMPERATURE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Known functional groups (open set — custom labels also accepted)
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_GROUPS = {
    "carboxylate":  "RCOOH → RCOO⁻ + H⁺",
    "ammonium":     "RNH₃⁺ → RNH₂ + H⁺  (aliphatic amine)",
    "imidazolium":  "ImH⁺  → Im   + H⁺",
    "guanidinium":  "RC(=NH)NH₂⁺ → RC(=NH)NH  (Arg side chain)",
    "phenol":       "ArOH  → ArO⁻ + H⁺",
    "thiol":        "RSH   → RS⁻  + H⁺",
    "indole":       "Trp indole NH",
    "phosphate":    "H₂PO₄⁻ → HPO₄²⁻ + H⁺",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepData:
    """One calibration data point: one ionisation step with computed + exp pKa."""
    compound:         str
    acid_name:        str
    base_name:        str
    acid_charge:      int
    functional_group: str
    pka_calc:         float   # ΔG_DFT / (RT ln10), uncorrected (~190–215)
    pka_exp:          float   # experimental value from pka_step annotation
    G_eff_acid:       float
    G_eff_base:       float


@dataclass
class LFERResult:
    """Fitted LFER for one (functional_group, acid_charge) combination."""
    functional_group:  str
    acid_charge:       int          # acid charge for this specific line
    description:       str
    n_points:          int
    slope:             float
    intercept:         float
    slope_se:          float
    intercept_se:      float
    slope_ci95:        tuple
    intercept_ci95:    tuple
    r2:                float
    rmse:              float
    mae:               float
    pka_calc_vals:     list[float]
    pka_exp_vals:      list[float]
    compound_names:    list[str]

# ──────────────────────────────────────────────────────────────────────────────
# Caching
# ──────────────────────────────────────────────────────────────────────────────

def _cache_path(msf_path: Path, out_dir: Path) -> Path:
    return out_dir / f"{msf_path.stem}_geff.json"


def load_cached_G_eff(msf_path: Path, out_dir: Path) -> Optional[dict[str, float]]:
    cache = _cache_path(msf_path, out_dir)
    if not cache.exists():
        return None
    try:
        data  = json.loads(cache.read_text())
        g_eff = {d["name"]: d["G_eff_Eh"] for d in data["microstates"]
                 if d.get("G_eff_Eh") is not None}
        if not g_eff:
            return None
        logging.info("  [%s] Loaded cached G_eff from %s", msf_path.stem, cache.name)
        return g_eff
    except (KeyError, json.JSONDecodeError) as e:
        logging.warning("  [%s] Cache read failed (%s) — will recompute.", msf_path.stem, e)
        return None


def save_G_eff_cache(msf_path: Path, microstates: list[Microstate],
                     method: str, out_dir: Path) -> None:
    cache = _cache_path(msf_path, out_dir)
    data  = {
        "compound": msf_path.stem,
        "method":   method,
        "microstates": [
            {
                "name":            ms.name,
                "charge":          ms.charge,
                "n_protons":       ms.n_protons,
                "G_eff_Eh":        ms.G_eff,
                "n_struct_found":  ms.n_struct_found,
                "n_struct_used":   ms.n_struct_used,
            }
            for ms in microstates
        ],
    }
    cache.write_text(json.dumps(data, indent=2))
    logging.info("  [%s] G_eff cached → %s", msf_path.stem, cache.name)

# ──────────────────────────────────────────────────────────────────────────────
# Per-compound calculation
# ──────────────────────────────────────────────────────────────────────────────

def compute_steps_for_compound(
        msf_path:     Path,
        out_dir:      Path,
        orca_binary:  str,
        nprocs:       int,
        nconf:        int,
        ewindow:      float,
        method:       str,
        do_tautomers: bool,
        temperature:  float,
        dry_run:      bool,
) -> list[StepData]:
    """
    Compute (or load from cache) G_eff for one training compound and
    return StepData for every step that has both pka_step and
    functional_group annotations on its base microstate.
    """
    method_kw = {
        "r2scan3c": "r2SCAN-3c", "r2scan-3c": "r2SCAN-3c",
        "pbe3c":    "PBE-3c",    "pbe-3c":    "PBE-3c",
    }.get(method.lower(), method)

    compound    = msf_path.stem
    microstates = parse_msf(msf_path)
    RT_ln10     = KB_EV * temperature / EH_TO_EV * LN10
    work_dir    = out_dir / f"work_{compound}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Check for usable annotations
    sorted_ms = sorted(microstates, key=lambda m: m.n_protons, reverse=True)
    usable_steps = [
        (a, b) for a, b in zip(sorted_ms[:-1], sorted_ms[1:])
        if b.pka_step is not None and b.functional_group is not None
    ]
    if not usable_steps:
        logging.warning(
            "[%s] No steps with both pka_step and functional_group — skipping.\n"
            "  Add 'pka_step = X' and 'functional_group = Y' to base microstates.",
            compound,
        )
        return []

    # ── Load cache or compute ─────────────────────────────────────────────────
    cached = load_cached_G_eff(msf_path, out_dir)
    if cached is not None:
        missing = [ms.name for ms in microstates if ms.name not in cached]
        if missing:
            logging.warning("[%s] Cache missing %s — recomputing.", compound, missing)
            cached = None
        else:
            for ms in microstates:
                ms.G_eff        = cached[ms.name]
                ms.n_struct_used = 1

    if cached is None:
        if dry_run:
            _mock_run(microstates, -0.447472, temperature)
        else:
            logging.info("[%s] Running ensemble calculations…", compound)
            for ms in microstates:
                run_ensemble(
                    ms=ms, work_dir=work_dir, orca_binary=orca_binary,
                    nprocs=nprocs, nconf=nconf, ewindow=ewindow,
                    method=method_kw, do_tautomers=do_tautomers,
                    temperature=temperature,
                )
        save_G_eff_cache(msf_path, microstates, method_kw, out_dir)

    # ── Extract step data ─────────────────────────────────────────────────────
    steps: list[StepData] = []
    for ms_acid, ms_base in usable_steps:
        if ms_acid.G_eff is None or ms_base.G_eff is None:
            logging.warning("[%s] Missing G_eff for %s→%s — skipping.",
                            compound, ms_acid.name, ms_base.name)
            continue
        dG    = ms_base.G_eff - ms_acid.G_eff
        pka_c = dG / RT_ln10
        fg    = ms_base.functional_group.strip().lower()
        steps.append(StepData(
            compound         = compound,
            acid_name        = ms_acid.name,
            base_name        = ms_base.name,
            acid_charge      = ms_acid.charge,
            functional_group = fg,
            pka_calc         = pka_c,
            pka_exp          = ms_base.pka_step,
            G_eff_acid       = ms_acid.G_eff,
            G_eff_base       = ms_base.G_eff,
        ))
        if fg not in KNOWN_GROUPS:
            logging.warning("  [%s] Unknown functional group '%s'. "
                            "Known groups: %s", compound, fg, list(KNOWN_GROUPS))
        logging.info(
            "  [%s] %s → %s  fg=%-14s  q=%+d  "
            "ΔG_DFT=%+.2f kcal/mol  pKa_exp=%.3f  "
            "(raw ΔG/RTln10=%+.1f, offset ~200 is normal — LFER corrects this)",
            compound, ms_acid.name, ms_base.name, fg, ms_acid.charge,
            dG * 627.509474, ms_base.pka_step, pka_c,
        )
    return steps

# ──────────────────────────────────────────────────────────────────────────────
# LFER fitting
# ──────────────────────────────────────────────────────────────────────────────

def lfer_key(fg: str, acid_charge: int) -> str:
    """Canonical string key for the lfer_params dict: e.g. 'carboxylate_q+0'."""
    return f"{fg}_q{acid_charge:+d}"


def fit_lfer(steps: list[StepData], functional_group: str, acid_charge: int,
             temperature: float = TEMPERATURE) -> Optional[LFERResult]:
    """
    Fit  pKa_exp = a × pKa_calc + b  for one (functional_group, acid_charge)
    combination.  Both must match exactly.

    N=1: slope fixed to 1, intercept only.
    N≥2: OLS with full statistics (SE, 95% CI via t-distribution).

    Returns None if no data points exist for this combination.
    """
    pts = [s for s in steps
           if s.functional_group == functional_group and s.acid_charge == acid_charge]
    if not pts:
        return None

    n    = len(pts)
    x    = np.array([s.pka_calc for s in pts])
    y    = np.array([s.pka_exp  for s in pts])
    fg   = functional_group
    desc = KNOWN_GROUPS.get(fg, "(custom functional group)")

    if n == 1:
        slope       = 1.0
        intercept   = float(y[0] - x[0])
        slope_se    = float("nan")
        int_se      = float("nan")
        slope_ci    = (float("nan"), float("nan"))
        int_ci      = (float("nan"), float("nan"))
        r2          = float("nan")
        residuals   = np.array([0.0])
        logging.warning(
            "  %s (q=%+d): 1 data point — slope=1, intercept only. "
            "Add more training compounds for a full fit.",
            lfer_key(fg, acid_charge), acid_charge,
        )
    else:
        res       = stats.linregress(x, y)
        slope     = float(res.slope)
        intercept = float(res.intercept)
        slope_se  = float(res.stderr)
        int_se    = float(res.intercept_stderr)
        r2        = float(res.rvalue ** 2)
        t95       = float(stats.t.ppf(0.975, df=n - 2))
        slope_ci  = (slope - t95 * slope_se, slope + t95 * slope_se)
        int_ci    = (intercept - t95 * int_se, intercept + t95 * int_se)
        residuals = y - (slope * x + intercept)

    return LFERResult(
        functional_group  = fg,
        acid_charge       = acid_charge,
        description       = desc,
        n_points          = n,
        slope             = slope,
        intercept         = intercept,
        slope_se          = slope_se,
        intercept_se      = int_se,
        slope_ci95        = slope_ci,
        intercept_ci95    = int_ci,
        r2                = r2,
        rmse              = float(np.sqrt(np.mean(residuals ** 2))),
        mae               = float(np.mean(np.abs(residuals))),
        pka_calc_vals     = x.tolist(),
        pka_exp_vals      = y.tolist(),
        compound_names    = [s.compound for s in pts],
    )

# ──────────────────────────────────────────────────────────────────────────────
# Output: lfer_params.json
# ──────────────────────────────────────────────────────────────────────────────

def save_lfer_params(results: dict[str, LFERResult], method: str,
                     out_path: Path) -> None:
    """
    Save LFER parameters to JSON keyed by 'functional_group_q<acid_charge>',
    e.g. 'carboxylate_q+0', 'ammonium_q+1', 'ammonium_q+0'.

    Structure:
    {
      "method": "r2SCAN-3c",
      "functional_groups": {
        "carboxylate_q+0": {
          "functional_group": "carboxylate",
          "acid_charge":      0,
          "description":      "RCOOH → RCOO⁻ + H⁺",
          "n_points":         6,
          "slope":            0.234,
          ...
        },
        "ammonium_q+0": { ... },
        "ammonium_q+1": { ... },
        ...
      }
    }
    """
    data: dict = {"method": method, "functional_groups": {}}
    for key, r in sorted(results.items()):
        data["functional_groups"][key] = {
            "functional_group": r.functional_group,
            "acid_charge":      r.acid_charge,
            "description":      r.description,
            "n_points":         r.n_points,
            "slope":            r.slope,
            "intercept":        r.intercept,
            "slope_se":         r.slope_se,
            "intercept_se":     r.intercept_se,
            "slope_ci95":       list(r.slope_ci95),
            "intercept_ci95":   list(r.intercept_ci95),
            "r2":               r.r2,
            "rmse":             r.rmse,
            "mae":              r.mae,
            "compounds":        r.compound_names,
            "pka_calc_vals":    r.pka_calc_vals,
            "pka_exp_vals":     r.pka_exp_vals,
        }
    out_path.write_text(json.dumps(data, indent=2))
    logging.info("LFER parameters saved → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Output: human-readable report
# ──────────────────────────────────────────────────────────────────────────────

def write_report(results: dict[str, LFERResult], method: str,
                 out_path: Path) -> None:
    def _fmt_ci(ci):
        if ci[0] != ci[0]:
            return "n/a  (need ≥2 data points)"
        return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
    def _fmt_se(v):
        return "n/a" if v != v else f"{v:.6f}"

    lines = [
        "=" * 72,
        "  pKa LFER Calibration Report  —  per (functional group, acid charge)",
        f"  DFT method: {method} / CPCM(Water) + GFN2-xTB RRHO",
        "=" * 72,
        "",
        "  Model:  pKa_pred = slope × pKa_calc + intercept",
        "          pKa_calc = ΔG_DFT / (RT ln10)   [uncorrected, ~190–215]",
        "",
        "  Each LFER line covers one functional group AT ONE CHARGE STATE.",
        "  The acid_charge is the formal charge of the MORE protonated species.",
        "",
    ]

    for key, r in sorted(results.items()):
        lines += [
            "─" * 72,
            f"  Key:              {key}",
            f"  Functional group: {r.functional_group}  "
            f"(acid charge q={r.acid_charge:+d} → {r.acid_charge-1:+d})",
            f"  Chemistry:        {r.description}",
            f"  N data points:    {r.n_points}"
            + ("  *** slope fixed=1 ***" if r.n_points == 1 else ""),
            "",
            f"  slope          =  {r.slope:+.6f}",
            f"    std error    =  {_fmt_se(r.slope_se)}",
            f"    95% CI       =  {_fmt_ci(r.slope_ci95)}",
            "",
            f"  intercept      =  {r.intercept:+.4f}  pKa units",
            f"    std error    =  {_fmt_se(r.intercept_se)}",
            f"    95% CI       =  {_fmt_ci(r.intercept_ci95)}",
            "",
            f"  R²             =  {'n/a' if r.r2 != r.r2 else f'{r.r2:.6f}'}",
            f"  RMSE           =  {r.rmse:.4f} pKa units",
            f"  MAE            =  {r.mae:.4f} pKa units",
            f"  pKa_calc range =  {min(r.pka_calc_vals):.2f} – {max(r.pka_calc_vals):.2f}"
            + (f"  (span {max(r.pka_calc_vals)-min(r.pka_calc_vals):.2f} units)"
               if r.n_points > 1 else ""),
            "",
            "  Training data:",
            f"    {'Compound':>22}  {'pKa_calc':>10}  {'pKa_exp':>8}"
            f"  {'pKa_pred':>9}  {'residual':>9}",
            "    " + "─" * 62,
        ]
        for cmpd, xv, yv in zip(r.compound_names, r.pka_calc_vals, r.pka_exp_vals):
            pred = r.slope * xv + r.intercept
            res  = yv - pred
            lines.append(
                f"    {cmpd:>22}  {xv:10.3f}  {yv:8.3f}"
                f"  {pred:9.3f}  {res:+9.4f}"
            )
        lines += ["", ""]

    # Coverage summary
    standard_keys = [
        lfer_key(fg, q) for fg in ["carboxylate", "ammonium", "imidazolium"]
        for q in [0, 1]
    ] + [lfer_key("phenol", 0), lfer_key("thiol", 0)]

    lines += [
        "=" * 72,
        "  Coverage Summary",
        "=" * 72, "",
        f"  {'Key':>20}  {'Status':>24}  {'N':>3}  {'RMSE':>6}  {'slope':>8}  {'R²':>8}",
        "  " + "─" * 76,
    ]
    all_keys = sorted(set(list(results.keys()) + standard_keys))
    for key in all_keys:
        if key in results:
            r    = results[key]
            r2s  = f"{r.r2:.4f}" if r.r2 == r.r2 else "  n/a"
            note = "*" if r.n_points == 1 else " "
            stat = f"✓ N={r.n_points}{note}"
            lines.append(
                f"  {key:>20}  {stat:>24}  {r.n_points:3d}  "
                f"{r.rmse:6.3f}  {r.slope:8.4f}  {r2s:>8}"
            )
        else:
            lines.append(f"  {key:>20}  {'✗ not calibrated':>24}   —      —         —        —")

    lines += ["", "  * slope=1 (single data point)", ""]
    out_path.write_text("\n".join(lines))
    logging.info("Report saved → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Output: calibration plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_calibration(results: dict[str, LFERResult], out_path: Path,
                     method: str) -> None:
    if not results:
        return

    n_plots = len(results)
    cols    = min(3, n_plots)
    rows    = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                             squeeze=False)

    for idx, (key, r) in enumerate(sorted(results.items())):
        ax = axes[idx // cols][idx % cols]
        x  = np.array(r.pka_calc_vals)
        y  = np.array(r.pka_exp_vals)

        ax.scatter(x, y, color="steelblue", s=70, zorder=3)
        for xi, yi, ci in zip(x, y, r.compound_names):
            ax.annotate(ci, (xi, yi), fontsize=6, textcoords="offset points",
                        xytext=(4, 2), color="dimgrey")

        if r.n_points >= 2:
            pad   = (x.max() - x.min()) * 0.08 + 0.1
            x_fit = np.linspace(x.min() - pad, x.max() + pad, 300)
            y_fit = r.slope * x_fit + r.intercept
            ax.plot(x_fit, y_fit, "k-", lw=1.5, zorder=2)

            n      = len(x)
            x_bar  = x.mean()
            se_fit = (r.rmse * np.sqrt(1/n + (x_fit - x_bar)**2
                       / max(np.sum((x - x_bar)**2), 1e-12)))
            t95    = float(__import__("scipy").stats.t.ppf(0.975, df=n - 2))
            ax.fill_between(x_fit, y_fit - t95 * se_fit, y_fit + t95 * se_fit,
                            alpha=0.12, color="steelblue")

        r2s   = f"R²={r.r2:.4f}" if r.r2 == r.r2 else "R²=n/a"
        title = (f"{key}  (N={r.n_points})\n"
                 f"slope={r.slope:.4f}  intercept={r.intercept:.2f}  "
                 f"{r2s}  RMSE={r.rmse:.3f}")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("pKa_calc  (ΔG_DFT / RT ln10)", fontsize=8)
        ax.set_ylabel("pKa_exp", fontsize=8)
        ax.grid(alpha=0.25)

    for idx in range(n_plots, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f"pKa LFER Calibration by Functional Group × Acid Charge\n"
                 f"{method} / CPCM(Water) + GFN2-xTB RRHO", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    logging.info("Calibration plot saved → %s", out_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Coverage check
# ──────────────────────────────────────────────────────────────────────────────

def check_coverage(lfer_json: Path) -> int:
    """
    Print coverage table for an existing lfer_params.json.
    Returns number of standard (fg, acid_charge) combinations not calibrated.
    """
    if not lfer_json.exists():
        print(f"ERROR: {lfer_json} does not exist.")
        return 99

    data  = json.loads(lfer_json.read_text())
    avail = data.get("functional_groups", {})

    print(f"LFER parameters file: {lfer_json}")
    print(f"DFT method: {data.get('method', 'unknown')}")
    print()
    print(f"  {'Key':>22}  {'Status':>26}  {'N':>3}  {'RMSE':>6}  {'slope':>8}  {'R²':>7}")
    print("  " + "─" * 78)

    # Standard keys needed for amino acids
    standard = [
        lfer_key("carboxylate", 0),   # neutral carboxylic acids
        lfer_key("carboxylate", 1),   # carboxylate on cation (e.g. His step1)  ← +2→+1 uses q_acid=2 actually
        lfer_key("ammonium", 0),      # α-NH3+ on zwitterion
        lfer_key("ammonium", 1),      # ε-NH3+ on cation
        lfer_key("imidazolium", 0),   # imidazole (neutral → anion)
        lfer_key("imidazolium", 1),   # imidazolium (cation → neutral)
        lfer_key("phenol", 0),
        lfer_key("thiol", 0),
    ]

    all_keys = sorted(set(list(avail.keys()) + standard))
    n_missing = 0
    for key in all_keys:
        if key in avail:
            v    = avail[key]
            n    = v["n_points"]
            r2s  = f"{v['r2']:.4f}" if isinstance(v['r2'], float) and v['r2']==v['r2'] else "  n/a"
            note = "*" if n == 1 else " "
            print(f"  {key:>22}  {'✓ calibrated':>26}  {n:3d}{note}  "
                  f"{v['rmse']:6.3f}  {v['slope']:8.4f}  {r2s:>7}")
        else:
            status = "✗ missing (standard)" if key in standard else "✗ missing"
            if key in standard:
                n_missing += 1
            print(f"  {key:>22}  {status:>26}   —      —         —       —")

    print()
    if n_missing == 0:
        print("  ✓ All standard (fg, charge) combinations are calibrated.")
    else:
        print(f"  ✗ {n_missing} standard combination(s) missing.")
    print()
    print("  * slope=1 (single data point); add more compounds for a full fit.")
    return n_missing


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _compute_geff_only(
        msf_path:    Path,
        out_dir:     Path,
        orca_binary: str,
        nprocs:      int,
        nconf:       int,
        ewindow:     float,
        method:      str,
        do_tautomers: bool,
        temperature: float,
        dry_run:     bool,
) -> None:
    """
    Run ORCA ensemble calculations for all microstates in msf_path and
    write the G_eff cache.  No pka_step or functional_group annotations
    are required — this is the right entry point for test/prediction
    compounds that are not part of the LFER training set.
    """
    compound    = msf_path.stem
    microstates = parse_msf(msf_path)
    work_dir    = out_dir / f"work_{compound}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Reuse cache if complete
    cached = load_cached_G_eff(msf_path, out_dir)
    if cached is not None:
        missing = [ms.name for ms in microstates if ms.name not in cached]
        if missing:
            logging.warning("[%s] Cache missing %s — recomputing.", compound, missing)
            cached = None
        else:
            logging.info("[%s] G_eff fully cached — skipping ORCA.", compound)
            for ms in microstates:
                ms.G_eff         = cached[ms.name]
                ms.n_struct_used = 1

    if cached is None:
        if dry_run:
            _mock_run(microstates, -0.447472, temperature)
        else:
            logging.info("[%s] Running ensemble calculations…", compound)
            for ms in microstates:
                run_ensemble(
                    ms=ms, work_dir=work_dir, orca_binary=orca_binary,
                    nprocs=nprocs, nconf=nconf, ewindow=ewindow,
                    method=method, do_tautomers=do_tautomers,
                    temperature=temperature,
                )
        save_G_eff_cache(msf_path, microstates, method, out_dir)

    # Report G_eff values
    for ms in sorted(microstates, key=lambda m: m.n_protons, reverse=True):
        g = ms.G_eff
        logging.info("  %-24s  nH=%d  z=%+d  G_eff=%s Eh",
                     ms.name, ms.n_protons, ms.charge,
                     f"{g:.8f}" if g is not None else "None")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        prog="pka_calibrate.py",
        description="Functional-group LFER calibration for pKa prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            # Initial calibration (training set — needs functional_group):
            python pka_calibrate.py acetic_acid.msf histidine.msf \\
                --orca /apps/software/orca/orca-6.1.0/orca --nprocs 12

            # Run ORCA for test compounds (no functional_group needed):
            python pka_calibrate.py test_set/*.msf \\
                --run-orca-only --outdir pka_results_test \\
                --orca /apps/software/orca/orca-6.1.0/orca --nprocs 12

            # Check which functional groups are calibrated:
            python pka_calibrate.py --check pka_results/lfer_params.json

            # Dry-run (no ORCA, for testing):
            python pka_calibrate.py acetic_acid.msf histidine.msf --dry-run
        """),
    )

    p.add_argument("--check", metavar="LFER_JSON",
                   help="Check coverage of existing lfer_params.json and exit.")
    p.add_argument("--run-orca-only", action="store_true",
                   help="Run ORCA ensemble calculations and cache G_eff without "
                        "fitting any LFER parameters. No functional_group annotation "
                        "needed. Use this for test/prediction compounds.")
    p.add_argument("msf_files", nargs="*", metavar="COMPOUND.msf")
    p.add_argument("--orca",         default="orca")
    p.add_argument("--nprocs",       type=int,   default=4)
    p.add_argument("--nconf",        type=int,   default=10)
    p.add_argument("--ewindow",      type=float, default=3.0)
    p.add_argument("--method",       default="r2scan3c")
    p.add_argument("--no-tautomers", action="store_true")
    p.add_argument("--temp",         type=float, default=TEMPERATURE)
    p.add_argument("--outdir",       default="pka_results")
    p.add_argument("--dry-run",      action="store_true")

    args = p.parse_args()

    if args.check:
        sys.exit(check_coverage(Path(args.check)))

    if not args.msf_files:
        p.error("Provide at least one COMPOUND.msf file, or --check LFER_JSON.")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_kw = {
        "r2scan3c": "r2SCAN-3c", "r2scan-3c": "r2SCAN-3c",
        "pbe3c":    "PBE-3c",    "pbe-3c":    "PBE-3c",
    }.get(args.method.lower(), args.method)

    # ── ORCA-only mode: just compute G_eff, no LFER fitting ───────────────────
    if args.run_orca_only:
        logging.info("=" * 62)
        logging.info("pka_calibrate.py  —  ORCA-only mode (G_eff caching)")
        logging.info("  Method:    %s / CPCM(Water) + GFN2-xTB RRHO", method_kw)
        logging.info("  Compounds: %d .msf files", len(args.msf_files))
        logging.info("  Output:    %s", out_dir)
        logging.info("  (No LFER fitting — no functional_group required)")
        logging.info("=" * 62)
        n_ok = 0
        for msf_file in args.msf_files:
            msf_path = Path(msf_file)
            if not msf_path.exists():
                logging.error("Not found: %s — skipping.", msf_path)
                continue
            logging.info("")
            logging.info("── %s ──", msf_path.stem)
            _compute_geff_only(
                msf_path=msf_path, out_dir=out_dir,
                orca_binary=args.orca, nprocs=args.nprocs,
                nconf=args.nconf, ewindow=args.ewindow,
                method=method_kw, do_tautomers=not args.no_tautomers,
                temperature=args.temp, dry_run=args.dry_run,
            )
            n_ok += 1
        print(f"\nDone — G_eff cached for {n_ok} compound(s) in {out_dir}/")
        print("Next step:")
        print(f"  python pka_gpr_calibrate.py {' '.join(args.msf_files)} \\")
        print(f"      --predict --gpr <training_outdir>/gpr_model.json \\")
        print(f"      --outdir {out_dir}")
        return

    logging.info("=" * 62)
    logging.info("pka_calibrate.py  —  functional-group LFER calibration")
    logging.info("  Method:    %s / CPCM(Water) + GFN2-xTB RRHO", method_kw)
    logging.info("  Compounds: %d .msf files", len(args.msf_files))
    logging.info("  Output:    %s", out_dir)
    logging.info("=" * 62)

    # ── Collect training data ─────────────────────────────────────────────────
    all_steps: list[StepData] = []
    for msf_file in args.msf_files:
        msf_path = Path(msf_file)
        if not msf_path.exists():
            logging.error("Not found: %s — skipping.", msf_path)
            continue
        logging.info("")
        logging.info("── %s ──", msf_path.stem)
        steps = compute_steps_for_compound(
            msf_path=msf_path, out_dir=out_dir,
            orca_binary=args.orca, nprocs=args.nprocs,
            nconf=args.nconf, ewindow=args.ewindow,
            method=args.method, do_tautomers=not args.no_tautomers,
            temperature=args.temp, dry_run=args.dry_run,
        )
        all_steps.extend(steps)
        logging.info("  → %d calibration step(s) contributed.", len(steps))

    if not all_steps:
        logging.error(
            "No calibration data collected.\n"
            "Ensure .msf files have both 'pka_step' and 'functional_group' "
            "on their base microstates."
        )
        sys.exit(1)

    # ── Fit LFER per (functional_group, acid_charge) ─────────────────────────
    logging.info("")
    logging.info("Fitting LFER per (functional group, acid charge)…")
    results: dict[str, LFERResult] = {}

    fg_charge_pairs = sorted(set((s.functional_group, s.acid_charge) for s in all_steps))
    for fg, q in fg_charge_pairs:
        r = fit_lfer(all_steps, fg, q, args.temp)
        if r:
            key        = lfer_key(fg, q)
            results[key] = r
            r2s = f"{r.r2:.4f}" if r.r2 == r.r2 else "n/a"
            logging.info(
                "  %-22s  N=%d  slope=%+.4f  intercept=%+.3f  RMSE=%.3f  R²=%s",
                key, r.n_points, r.slope, r.intercept, r.rmse, r2s,
            )

    # ── Save outputs ──────────────────────────────────────────────────────────
    lfer_path  = out_dir / "lfer_params.json"
    report_path = out_dir / "lfer_report.txt"
    plot_path  = out_dir / "calibration_plot.png"

    save_lfer_params(results, method_kw, lfer_path)
    write_report(results, method_kw, report_path)
    plot_calibration(results, plot_path, method_kw)

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  LFER Calibration Summary")
    print(f"  {method_kw} / CPCM(Water) + GFN2-xTB RRHO")
    print("=" * 65)
    print()
    print(f"  {'Key':>22}  {'N':>3}  {'slope':>8}  {'intercept':>11}"
          f"  {'RMSE':>6}  {'R²':>7}")
    print("  " + "─" * 63)
    for key, r in sorted(results.items()):
        r2s  = f"{r.r2:.4f}" if r.r2 == r.r2 else "   n/a"
        note = " *" if r.n_points == 1 else "  "
        print(f"  {key:>22}  {r.n_points:3d}  {r.slope:+8.4f}  "
              f"{r.intercept:+11.3f}  {r.rmse:6.3f}  {r2s:>7}{note}")
    print()
    print("  * slope=1.0 (single data point)")
    print()

    standard = [
        lfer_key("carboxylate", 0), lfer_key("ammonium", 0),
        lfer_key("ammonium", 1),    lfer_key("imidazolium", 1),
    ]
    missing = [k for k in standard if k not in results]
    if missing:
        print(f"  WARNING: core keys not yet calibrated: {missing}")
    else:
        print("  ✓ Core combinations calibrated.")
    print()
    print(f"  Output: {lfer_path}")
    print(f"          {report_path}")
    print(f"          {plot_path}")
    print()
    print("  Use with pka_calc.py:")
    print("    python pka_calc.py molecule.msf \\")
    print("        --lfer pka_results/lfer_params.json --orca ...")


if __name__ == "__main__":
    main()
