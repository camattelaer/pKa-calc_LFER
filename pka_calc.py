#!/usr/bin/env python3
"""
pka_calc.py  —  Macroscopic pKa from conformational/tautomeric ensembles
=========================================================================
Predicts pH-dependent protonation state populations and macroscopic pKa
values for an unknown compound from first principles, without any
experimental pKa input.

Theory
------
For each protonation microstate i (defined by formal charge and number of
titratable protons n_H,i), the free energy in solution is:

    G_eff,i = −kT · ln Σ_c exp(−G_composite,c / kT)          (Eq. 1)

where the sum runs over all unique conformers and tautomers of microstate i,
and the composite free energy of structure c is:

    G_composite,c = E_DFT,c(CPCM) + G_RRHO,c(GFN2) − E_GFN2,c(ALPB) (Eq. 2)

This replaces the GFN2-xTB electronic energy with a DFT single-point energy
while retaining the GFN2-xTB thermal (RRHO) correction.  The solvation
model switches from ALPB (GFN2) to CPCM (DFT), consistent with the DFT
energy scale.

The proton free energy in solution is obtained isodesmically from one
reference compound with a known experimental pKa (Eq. 3):

    G*(H+,aq) = pKa_ref × RT ln10  −  G_eff(A_ref)  +  G_eff(HA_ref)

This eliminates the ~13 pKa unit systematic error of the absolute
thermodynamic cycle at r2SCAN-3c/CPCM level.  The isodesmic correction
works for ALL protonation steps of a polyprotic target, including all
charge transitions, because the DFT/CPCM error in G*(H+) is a uniform
offset rather than the charge-dependent error seen with GFN2-xTB.

The pH-dependent Boltzmann populations follow arXiv:2604.00841:

    G_i^corr = G_eff,i - n_H,i * G*(H+,aq)
    P_i(pH)  = softmax(-beta * [G_i^corr - G_ref + n_H,i * RT*ln10*pH])  (Eq. 4)

Levels of theory
----------------
  Structure search  :  GFN2-xTB / ALPB(Water)     via ORCA GOAT
  Tautomer search   :  GFN2-xTB / ALPB(Water)     via ORCA GOAT (TautSearch)
  Thermal correction:  GFN2-xTB / ALPB(Water)     OPT + FREQ
  Electronic energy :  r2SCAN-3c / CPCM(Water)    SP  (default)
                    or PBE-3c   / CPCM(Water)    SP  (--method pbe3c, faster)
  H-atom SP         :  r2SCAN-3c / gas phase

Input file format (.msf)
------------------------
  # Comments start with #.  Blank lines are ignored.

  [microstate]
  name         = Neutral           # label used in filenames and plots
  charge       = 0                 # formal charge
  multiplicity = 1                 # spin multiplicity (1 = closed-shell)
  n_protons    = 1                 # titratable H count vs fully-deprotonated
                                   # n_protons = 0 is the reference microstate

  xyz =
  C   0.000  0.000  0.000
  O   1.200  0.000  0.000
  H   1.800  0.900  0.000

  Rules:
  - Exactly one microstate must have n_protons = 0.
  - n_protons must be consecutive integers: 0, 1, 2, ...
  - pka_step (optional): experimental pKa for the ionisation step that
    produces this microstate.  Required on the BASE microstate of each
    reference compound pair (not needed in target .msf files).

Usage
-----
  # Generate .msf template from xyz files:
  python pka_calc.py --make-msf neutral.xyz anion.xyz --out molecule.msf

  # Full calculation — reference required for accurate results:
  python pka_calc.py molecule.msf --ref reference.msf \
      --orca /path/to/orca --nprocs 12

  # Faster with PBE-3c:
  python pka_calc.py molecule.msf --ref reference.msf \
      --orca /path/to/orca --nprocs 12 --method pbe3c

  # Skip tautomer search:
  python pka_calc.py molecule.msf --ref reference.msf --orca ... --no-tautomers

  # Dry-run (no ORCA, pipeline test):
  python pka_calc.py molecule.msf --ref reference.msf --dry-run

The reference .msf file contains one microstate pair with a known pKa_step
value (e.g. acetic acid, pKa = 4.756).  A single monoprotic reference
corrects all ionisation steps of any target at DFT/CPCM level.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────────────
KB_EV       = 8.617333262e-5    # eV / K
EH_TO_EV    = 27.211386245      # Hartree → eV
EH_TO_KCAL  = 627.509474        # Hartree → kcal/mol
KCAL_TO_EH  = 1.0 / EH_TO_KCAL
LN10        = np.log(10.0)
TEMPERATURE = 298.15            # K

# Absolute cycle constants (Eq. 3)
IE_HYDROGEN_EH    = 13.598434599702 / EH_TO_EV   # NIST ionisation energy of H
DG_SOLV_PROTON_EH = -263.98 * KCAL_TO_EH          # Fawcett 2004

# ──────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ──────────────────────────────────────────────────────────────────────────────

class SCFDivergenceError(RuntimeError):
    """
    Raised when an ORCA/xTB output contains NaN values in the SCF iteration
    block, indicating the self-consistent field procedure diverged.

    This most commonly occurs for highly charged anions (charge ≤ −2) where
    the GFN2-xTB/GBSA combination is numerically unstable for certain starting
    geometries from GOAT.  The electronic energy and Gibbs correction from such
    a run are unreliable and must not be used in the Boltzmann ensemble.

    The caller (run_ensemble) catches this and skips the affected conformer,
    continuing with any remaining structures that converged successfully.
    """


class MissingThermoError(RuntimeError):
    """
    Raised when an ORCA/xTB OPT+FREQ output does not contain a Gibbs free
    energy block — meaning the thermochemistry section was never printed,
    typically because the geometry optimisation did not converge or the FREQ
    step was not reached.

    Unlike SCFDivergenceError (NaN in SCF), this case has a valid electronic
    energy E_elec but no G_RRHO correction.  Using E_elec alone as a proxy
    for G introduces an error of ~10–23 pKa units (the RRHO correction for
    small molecules), which completely invalidates the pKa_calc value.

    The caller (run_ensemble) catches this and retries with robust SCF
    settings before giving up on the conformer.
    """

@dataclass
class Structure:
    """One conformer or tautomer of a microstate after search and optimisation."""
    label:        str
    xyz_block:    str             # Cartesian coordinates (body, no header)
    G_xtb_aq:    Optional[float] = None   # G(GFN2-xTB, ALPB)  [Eh]
    E_xtb_aq:    Optional[float] = None   # E_elec(GFN2-xTB, ALPB) [Eh]
    E_dft_aq:    Optional[float] = None   # E(DFT, CPCM) [Eh]
    G_composite: Optional[float] = None   # Eq. 2 [Eh]


@dataclass
class Microstate:
    """
    One protonation state of the target molecule.
    n_protons counts titratable protons relative to the fully-deprotonated
    reference (which must have n_protons = 0).
    """
    name:         str
    xyz_block:    str
    charge:       int
    multiplicity: int
    n_protons:    int
    pka_step:         Optional[float] = None   # experimental pKa (reference/training files)
    functional_group: Optional[str]   = None   # e.g. "carboxylate", "ammonium", ...

    structures:    list[Structure] = field(default_factory=list, repr=False)
    G_eff:         Optional[float] = field(default=None, repr=False)
    n_struct_found: int = 0
    n_struct_used:  int = 0

# ──────────────────────────────────────────────────────────────────────────────
# .msf parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_msf(path: Path) -> list[Microstate]:
    """
    Parse a .msf microstate file.
    Returns microstates sorted highest n_protons first.
    """
    text   = Path(path).read_text()
    blocks = re.split(r'\[microstate\]', text, flags=re.IGNORECASE)
    if len(blocks) < 2:
        raise ValueError(f"{path}: no [microstate] blocks found.")

    result: list[Microstate] = []
    for raw in blocks[1:]:
        meta:    dict  = {}
        xyz_buf: list  = []
        in_xyz         = False
        for line in raw.splitlines():
            s = line.strip()
            if not in_xyz and (not s or s.startswith('#')):
                continue
            if in_xyz:
                if s and not s.startswith('#'):
                    xyz_buf.append(line)
            else:
                if re.match(r'xyz\s*=', s, re.IGNORECASE):
                    in_xyz = True
                elif '=' in s:
                    k, _, v = s.partition('=')
                    v = v.split('#')[0].strip()
                    if v:
                        meta[k.strip().lower().replace(' ', '_')] = v

        if not meta and not xyz_buf:
            continue
        if 'n_protons' not in meta:
            raise ValueError(f"{path}: block missing 'n_protons': {meta}")
        xyz = '\n'.join(xyz_buf).strip()
        if not xyz:
            raise ValueError(f"{path}: '{meta.get('name','?')}' has no coordinates.")

        result.append(Microstate(
            name             = meta.get('name', f'ms{len(result)}'),
            xyz_block        = xyz,
            charge           = int(meta.get('charge', 0)),
            multiplicity     = int(meta.get('multiplicity', 1)),
            n_protons        = int(meta['n_protons']),
            pka_step         = float(meta['pka_step'])         if 'pka_step'         in meta else None,
            functional_group = meta['functional_group'].strip().lower() if 'functional_group' in meta else None,
        ))

    if not result:
        raise ValueError(f"{path}: no microstates parsed.")

    n_vals = sorted(set(m.n_protons for m in result))
    if n_vals[0] != 0:
        raise ValueError(f"{path}: no microstate with n_protons=0.")
    if n_vals != list(range(len(n_vals))):
        raise ValueError(f"{path}: n_protons must be 0,1,2,…  Got {n_vals}")

    result.sort(key=lambda m: m.n_protons, reverse=True)
    return result


def derive_G_proton_per_step(
        ref_microstates: list[Microstate],
        temperature:     float = TEMPERATURE,
) -> dict[int, float]:
    """
    Derive G*(H⁺,aq) per ionisation step from a reference compound,
    keyed by the ACID CHARGE of each step.

    For each adjacent pair (acid, base) in the reference that has a
    pka_step annotation on the base microstate:

        G*(H+)_{q_acid} = pKa × RT × ln10  −  G_eff(base)  +  G_eff(acid)

    The result is stored as {q_acid: G*(H+)} so that the same dict can
    supply G*(H+) for any target step with the same charge transition type.

    WHY PER ACID CHARGE:
    Implicit solvent models (CPCM, ALPB) have solvation energies that scale
    as q² (Born), so the systematic error in G*(H+) differs substantially
    between charge transitions +2→+1, +1→0, 0→-1, etc.  A reference
    compound with the same acid charge as the target step provides the
    correct cancellation.  Differences between transition types are
    ~6-10 kcal/mol at r2SCAN-3c/CPCM level.

    Raises ValueError if no pka_step annotations are found.
    """
    RT_ln10    = KB_EV * temperature / EH_TO_EV * LN10
    sorted_ref = sorted(ref_microstates, key=lambda m: m.n_protons, reverse=True)

    G_H_by_charge: dict[int, float] = {}
    for ms_acid, ms_base in zip(sorted_ref[:-1], sorted_ref[1:]):
        if ms_base.pka_step is None:
            continue
        if ms_acid.G_eff is None or ms_base.G_eff is None:
            raise RuntimeError(
                f"Reference '{ms_acid.name}' or '{ms_base.name}' has no G_eff."
            )
        G_H_k  = ms_base.pka_step * RT_ln10 - ms_base.G_eff + ms_acid.G_eff
        q_acid = ms_acid.charge
        G_H_by_charge[q_acid] = G_H_k
        dG_raw = (ms_base.G_eff - ms_acid.G_eff) * EH_TO_KCAL
        logging.info(
            "  ref step (%s → %s)  q_acid=%+d  pKa_exp=%.3f"
            "  ΔG_raw=%+.2f kcal/mol  G*(H+)=%.6f Eh",
            ms_acid.name, ms_base.name, q_acid,
            ms_base.pka_step, dG_raw, G_H_k,
        )

    if not G_H_by_charge:
        raise ValueError(
            "No pka_step values found in reference .msf.  "
            "Add 'pka_step = <experimental_pKa>' to the base microstate(s)."
        )
    return G_H_by_charge


def map_G_proton_to_target(
        G_H_by_charge:  dict[int, float],
        tgt_microstates: list[Microstate],
) -> list[float]:
    """
    Map reference G*(H+) values to target ionisation steps by matching
    the acid charge of each step.

    For a target step (acid → base) where acid.charge = q:
      - If q is in G_H_by_charge: use G_H_by_charge[q] directly.
      - Otherwise: use the entry with the nearest charge and warn.

    Returns a list of G*(H+) values, one per target step, ordered from
    the highest-pKa step (n_protons_max → n_protons_max-1) downward.
    This is the ORDER expected by compute_populations (cumulative sum
    from the end of the list for each microstate's correction).
    """
    sorted_tgt = sorted(tgt_microstates, key=lambda m: m.n_protons, reverse=True)
    available  = sorted(G_H_by_charge.keys())
    result     = []

    for ms_acid, ms_base in zip(sorted_tgt[:-1], sorted_tgt[1:]):
        q = ms_acid.charge
        if q in G_H_by_charge:
            G_H = G_H_by_charge[q]
            logging.info(
                "  tgt step (%s → %s)  q_acid=%+d: exact match → G*(H+)=%.6f Eh",
                ms_acid.name, ms_base.name, q, G_H,
            )
        else:
            nearest_q = min(available, key=lambda x: abs(x - q))
            G_H = G_H_by_charge[nearest_q]
            logging.warning(
                "  tgt step (%s → %s)  q_acid=%+d: no reference for this charge."
                "  Using nearest (q=%+d).  Accuracy reduced for this step.",
                ms_acid.name, ms_base.name, q, nearest_q,
            )
        result.append(G_H)
    return result


def compute_populations(
        microstates: list[Microstate],
        G_proton:    float | list[float],
        pH_array:    np.ndarray,
        temperature: float = TEMPERATURE,
) -> np.ndarray:
    """
    Boltzmann populations vs pH.

    G_proton may be:
      - a single float: the same G*(H+) is used for all ionisation steps
        (correct only when all steps have the same charge transition type)
      - a list of floats: one per ionisation step, ordered step-1-first
        (step-1 = highest-pKa transition = n_protons_max → n_protons_max-1)
        This is the correct treatment for polyprotic molecules spanning
        multiple charge states.

    For microstate with n_H titratable protons:
        G_corr(i) = G_eff(i) − Σ_{k = N-n_H+1 to N} G*(H+)_k

    The sum covers the LAST n_H steps (highest-pKa steps), i.e.:
        cumulative[n_H] = sum(G_proton_list[N - n_H :])

    Returns populations array of shape (N_microstates, N_pH).
    """
    kT    = KB_EV * temperature / EH_TO_EV
    G_eff = np.array([ms.G_eff for ms in microstates])
    n_H   = np.array([ms.n_protons for ms in microstates], dtype=float)
    N     = int(n_H.max())   # number of ionisation steps

    # Build per-step list and cumulative corrections
    if isinstance(G_proton, (int, float)):
        G_list = [float(G_proton)] * N
    else:
        G_list = list(G_proton)
        if len(G_list) < N:
            G_list = G_list + [G_list[-1]] * (N - len(G_list))
        G_list = G_list[:N]

    # cumulative[n] = sum of the LAST n G*(H+) steps
    # (the n highest-pKa steps, which correspond to the protons present in
    #  a microstate with n_H = n titratable protons)
    cumulative = [0.0] * (N + 1)
    for n in range(1, N + 1):
        cumulative[n] = sum(G_list[N - n:])

    G_corr   = np.array([ms.G_eff - cumulative[ms.n_protons] for ms in microstates])
    ref_idx  = next(i for i, ms in enumerate(microstates) if ms.n_protons == 0)
    G_ref    = G_corr[ref_idx]

    beta_dG0 = (G_corr - G_ref) / kT
    beta_dG  = beta_dG0[:, None] + n_H[:, None] * LN10 * pH_array[None, :]

    log_w = -beta_dG
    log_w -= log_w.max(axis=0, keepdims=True)
    w = np.exp(log_w)
    return w / w.sum(axis=0, keepdims=True)



# ──────────────────────────────────────────────────────────────────────────────
# ORCA I/O primitives
# ──────────────────────────────────────────────────────────────────────────────

def _run_orca(inp_path: Path, orca_binary: str) -> Path:
    """Run ORCA.  inp_path must be a file; runs with cwd = its parent."""
    job_dir  = inp_path.parent
    out_path = job_dir / inp_path.with_suffix('.out').name
    logging.info("  ▶  cd %s && %s %s", job_dir, orca_binary, inp_path.name)
    proc = subprocess.run(
        [orca_binary, inp_path.name],
        cwd=job_dir, capture_output=True, text=True,
    )
    out_path.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ORCA failed (rc={proc.returncode}) — see {out_path}"
        )
    return out_path


def _parse_sp_energy(out_path: Path) -> float:
    """Parse last 'FINAL SINGLE POINT ENERGY' from an ORCA output."""
    txt     = out_path.read_text(errors='replace')
    matches = re.findall(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)', txt)
    if not matches:
        raise RuntimeError(f"Cannot parse SP energy from {out_path}")
    return float(matches[-1])


def _parse_gibbs_and_sp(out_path: Path) -> tuple[float, float]:
    """
    Parse (G_total, E_elec) from an ORCA OPT+FREQ output.
    G_total = E_elec + ZPE + thermal corrections (RRHO).

    Handles all ORCA 6.x GFN2-xTB output variants:
      - 'Final Gibbs free energy         ...   -14.439 Eh'
      - 'Total Free Energy               ...   -14.439 Eh'  (older ORCA)
      - 'G(T)     =  -14.43986268 Eh'                       (xTB thermo block)

    Raises SCFDivergenceError if NaN values appear in the SCF iteration table.
    This indicates the xTB SCF diverged (common for highly charged anions at
    certain GOAT geometries). The caller should skip this conformer entirely.

    If the Gibbs block is absent but E_elec is present and SCF converged (e.g.
    OPT converged but FREQ was skipped), falls back to E_elec with G_RRHO = 0
    and logs a warning.
    """
    txt = out_path.read_text(errors='replace')

    # ── Detect SCF NaN divergence ─────────────────────────────────────────────
    # Pattern: SCF iteration lines contain "NaN" in energy/gradient columns.
    # We look for at least two consecutive NaN iteration lines to avoid false
    # positives from "NaN" appearing elsewhere in ORCA output.
    nan_iters = re.findall(
        r'^\s*\d+\s+NaN\s+NaN\s+NaN', txt, re.MULTILINE
    )
    if len(nan_iters) >= 2:
        raise SCFDivergenceError(
            f"{out_path.name}: SCF diverged to NaN ({len(nan_iters)} NaN "
            f"iterations). This structure will be excluded from the ensemble. "
            f"Common cause: highly charged anion with unstable GFN2-xTB/GBSA "
            f"starting geometry."
        )

    # ── Check for imaginary frequencies ───────────────────────────────────────
    # Distinguish numerical noise from genuine saddle points:
    #
    #   |ω| < 50 cm⁻¹  → numerical artefact of the finite-difference Hessian
    #                    or a soft torsional mode slightly below zero.
    #                    Gibbs correction is negligibly affected (kT ≈ 207 cm⁻¹).
    #                    WARN and INCLUDE — excluding would silently drop single-
    #                    conformer compounds (e.g. rigid acetate with ω = −24 cm⁻¹).
    #
    #   |ω| ≥ 50 cm⁻¹  → genuine saddle point. The geometry is not a minimum;
    #                    the Gibbs energy is thermodynamically unreliable.
    #                    EXCLUDE via MissingThermoError.
    #
    # Note: the captopril/valsartan pKa_calc = −86538 catastrophe was caused
    # by wrong atom counts in the MSF files, NOT by imaginary frequencies.
    # All imaginary modes in that case were −4 to −13 cm⁻¹ numerical noise.
    IMAG_FREQ_THRESHOLD = 50.0   # cm⁻¹
    imag_vals = [float(f) for f in
                 re.findall(r'^\s*\d+:\s+([-]\d+\.\d+)\s+cm', txt, re.MULTILINE)
                 if float(f) < 0]
    if imag_vals:
        largest_imag = min(imag_vals)   # most negative
        if abs(largest_imag) >= IMAG_FREQ_THRESHOLD:
            raise MissingThermoError(
                f"{out_path.name}: imaginary frequency {largest_imag:.1f} cm⁻¹ "
                f"(|ω| ≥ {IMAG_FREQ_THRESHOLD} cm⁻¹ threshold) — genuine saddle "
                f"point, excluding from ensemble."
            )
        else:
            logging.warning(
                "    %s: imaginary frequency %.1f cm⁻¹ (|ω| < %.0f cm⁻¹) — "
                "numerical noise, including in ensemble.",
                out_path.name, largest_imag, IMAG_FREQ_THRESHOLD
            )

    # ── Parse Gibbs free energy ────────────────────────────────────────────────
    G = None
    for pat in [
        r'Final Gibbs free energy\s*\.+\s*([-\d.]+)\s*Eh',
        r'Total Free Energy\s*\.+\s*([-\d.]+)\s*Eh',
        r'G\(T\)\s*=\s*([-\d.]+)\s*Eh',
        r'Total enthalpy\s*\.+\s*([-\d.]+)\s*Eh',
    ]:
        m = re.search(pat, txt)
        if m:
            G = float(m.group(1))
            break

    # ── Parse electronic energy ────────────────────────────────────────────────
    matches_E = re.findall(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)', txt)
    if not matches_E:
        # No energy at all — ORCA did not reach the SCF step.
        # This is different from MissingThermoError (SCF ran but FREQ didn't).
        # Common causes: invalid keyword (e.g. VeryTightOpt with GFN2-xTB),
        # ORCA licence error, or missing xTB parameter files.
        # Check the .out file for ERROR or ORCA_FATAL lines for diagnosis.
        n_lines = len(txt.splitlines())
        hint = ""
        if "error" in txt.lower() or "fatal" in txt.lower():
            for line in txt.splitlines():
                if "error" in line.lower() or "fatal" in line.lower():
                    hint = f" ORCA says: {line.strip()[:120]}"
                    break
        raise RuntimeError(
            f"Cannot parse electronic energy from {out_path} "
            f"({n_lines} lines).{hint} "
            f"ORCA likely crashed before reaching the SCF — check the .out "
            f"file for ERROR/FATAL lines. Common causes: invalid keyword for "
            f"this method (e.g. VeryTightOpt with GFN2-xTB), licence issues, "
            f"or missing parameter files."
        )
    E = float(matches_E[-1])

    if G is None:
        raise MissingThermoError(
            f"{out_path.name}: Gibbs energy block not found — the OPT+FREQ "
            f"did not complete thermochemistry (OPT may not have converged). "
            f"E_elec = {E:.8f} Eh.  Will retry with robust SCF settings."
        )

    return G, E


def _read_optimised_xyz(freq_out: Path) -> Optional[str]:
    """
    Read the optimised geometry from the .xyz file written by ORCA OPT.
    Returns the body (no header) or None if the file is missing.
    """
    xyz_path = freq_out.with_suffix('.xyz')
    if not xyz_path.exists():
        # Try alternative naming (ORCA writes <basename>.xyz)
        candidates = list(freq_out.parent.glob('*.xyz'))
        if not candidates:
            return None
        xyz_path = candidates[0]
    lines = xyz_path.read_text().splitlines()
    try:
        n_at = int(lines[0].strip())
        return '\n'.join(lines[2: 2 + n_at])
    except (ValueError, IndexError):
        return None


def _parse_goat_ensemble_xyz(goat_dir: Path) -> list[str]:
    """
    Parse the finalensemble.xyz file written by GOAT.
    Returns list of xyz body strings, in GOAT's energy-ranked order.
    """
    candidates = list(goat_dir.glob('*.finalensemble.xyz'))
    if not candidates:
        return []
    text   = candidates[0].read_text()
    lines  = text.splitlines()
    bodies = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if re.match(r'^\d+$', stripped):
            n_at = int(stripped)
            body = '\n'.join(lines[i + 2: i + 2 + n_at])
            bodies.append(body)
            i += 2 + n_at
        else:
            i += 1
    return bodies

# ──────────────────────────────────────────────────────────────────────────────
# ORCA input writers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_xyz(xyz_body: str) -> str:
    """Strip each atom line so the xyz block has no leading whitespace."""
    return '\n'.join(
        line.strip() for line in xyz_body.splitlines() if line.strip()
    )


def _goat_inp(ms: Microstate, job_dir: Path, nprocs: int,
              solvent: str, do_tautomers: bool) -> Path:
    """Write ORCA GFN2-xTB GOAT conformer-search input."""
    xyz  = _clean_xyz(ms.xyz_block)
    inp  = job_dir / f"{ms.name}_goat.inp"
    inp.write_text(
        f"! GFN2-XTB {solvent} GOAT TightSCF\n"
        f"%pal nprocs {nprocs} end\n"
        f"* xyz {ms.charge} {ms.multiplicity}\n"
        f"{xyz}\n"
        f"*\n"
    )
    return inp


def _freq_inp(ms: Microstate, xyz_body: str, job_dir: Path,
              label: str, nprocs: int, solvent: str,
              robust: bool = False, very_robust: bool = False) -> Path:
    """
    Write ORCA GFN2-xTB OPT+FREQ input.

    robust=True activates convergence aids for difficult geometries:
      - Electronic temperature raised to 3000 K (etemp 3000) so fractional
        occupation smears the frontier and prevents sudden orbital crossings
        that destabilise the SCF.
      - Broyden damping tightened to 0.7 (default 0.4) to prevent overshoot.
      - SCF iteration limit raised to 500.
      - Geometry optimisation iteration limit raised to 500 (default 50).
        This is the primary fix for floppy zwitterions (serine, threonine, etc.)
        where the OPT hits the 50-cycle ORCA default and exits before FREQ runs.
        The MissingThermoError is then caught and the retry uses this input —
        but with standard robust=False the geometry cycle limit stays at 50,
        so retry with the same geometry fails identically.
      - SlowConv convergence strategy instead of the default.

    very_robust=True (implies robust=True) escalates further:
      - Electronic temperature raised to 5000 K — broader smearing for rigid
        aromatic cations where 3000 K is insufficient (e.g. 3-chloroanilinium).
      - Broyden damping tightened to 0.9.
      - VerySlowConv strategy.
      This is the last-resort level before a conformer is permanently skipped.
    """
    xyz = _clean_xyz(xyz_body)
    inp = job_dir / f"{label}_freq.inp"

    if very_robust:
        scf_block = (
            f"%scf\n"
            f"  maxiter 500\n"
            f"  damp 0.9\n"
            f"end\n"
            f"%geom\n"
            f"  MaxIter 500\n"
            f"end\n"
            f"%xtb\n"
            f"  etemp 5000\n"
            f"end\n"
        )
        conv_kw = "VerySlowConv"
    elif robust:
        scf_block = (
            f"%scf\n"
            f"  maxiter 500\n"
            f"  damp 0.7\n"
            f"end\n"
            f"%geom\n"
            f"  MaxIter 500\n"
            f"end\n"
            f"%xtb\n"
            f"  etemp 3000\n"
            f"end\n"
        )
        conv_kw = "SlowConv"
    else:
        scf_block = ""
        conv_kw   = "TightSCF"  # VeryTightOpt removed: not valid for GFN2-xTB

    inp.write_text(
        f"! GFN2-XTB {solvent} OPT FREQ {conv_kw}\n"
        f"%pal nprocs {nprocs} end\n"
        f"{scf_block}"
        f"%output\n"
        f"  Print[P_Mulliken] 1\n"
        f"end\n"
        f"* xyz {ms.charge} {ms.multiplicity}\n"
        f"{xyz}\n"
        f"*\n"
    )
    return inp


def _dft_sp_inp(ms: Microstate, xyz_body: str, job_dir: Path,
                label: str, nprocs: int, method: str, solvent: str,
                slow_conv: bool = False) -> Path:
    """
    Write ORCA DFT single-point input (r2SCAN-3c / CPCM).

    slow_conv=True adds SlowConv keyword for difficult SCF convergence.
    Used as a retry when the standard DFT SP fails.

    CHELPG: charges fitted to the molecular electrostatic potential on the
    solvent-accessible surface (COSMO VDW radii, grid 0.3 Å, Rmax 2.8 Å).
    Unlike element-summed Hirshfeld charges, CHELPG per-atom charges encode
    the full ESP including inductive + mesomeric + through-space effects from
    ALL surrounding groups (ionisable or not).  Key for drug-like molecules
    where remote substituents (CF3, amide C=O, COO-) modulate the pKa.
    """
    xyz = _clean_xyz(xyz_body)
    inp = job_dir / f"{label}_dft.inp"
    conv_kw = "SlowConv" if slow_conv else "TightSCF"
    inp.write_text(
        f"! {method} {solvent} {conv_kw}\n"
        f"%pal nprocs {nprocs} end\n"
        f"%output\n"
        f"  Print[P_Hirshfeld] 1\n"
        f"  Print[P_Mulliken]  1\n"
        f"end\n"
        f"%chelpg\n"
        f"  GRID 0.3\n"
        f"  RMAX 2.8\n"
        f"  VDWRADII COSMO\n"
        f"  DIPOLE TRUE\n"
        f"end\n"
        f"* xyz {ms.charge} {ms.multiplicity}\n"
        f"{xyz}\n"
        f"*\n"
    )
    return inp


def _dft_inp_is_current(inp_path: Path, ms: Microstate, xyz_body: str,
                         nprocs: int, method: str, solvent: str) -> bool:
    """
    Check whether an existing DFT calculation was run with the current input
    template by extracting the verbatim input ORCA embedded in the .out file.

    ORCA echoes the exact input it ran between:
        NAME = <filename>
        |  1> line1
        |  2> line2
        ****END OF INPUT****

    We extract and strip the `| N>` prefixes, then check for the canonical
    markers of the current _dft_sp_inp template — specifically the %chelpg
    block with GRID 0.3 / RMAX 2.8 / VDWRADII COSMO.

    Returns True  → output was generated with current input; no rerun needed.
    Returns False → output predates %chelpg addition, or .out does not exist.

    We read from the .out file (not the .inp file) because ORCA may overwrite
    or the .inp may be absent, but the .out always contains the verbatim input
    that was actually executed.
    """
    # Derive .out path from .inp path (same stem, different suffix)
    out_path = inp_path.with_suffix('.out')
    if not out_path.exists():
        return False

    txt = out_path.read_text(errors='replace')

    # Extract the embedded input block
    m = re.search(r'NAME\s*=\s*\S+\s*\n(.*?)\*\*\*\*END OF INPUT\*\*\*\*',
                  txt, re.DOTALL)
    if not m:
        return False

    # Strip | N> prefixes to recover the raw input text
    lines = [re.sub(r'^\|\s*\d+>\s?', '', l) for l in m.group(1).splitlines()]
    embedded = '\n'.join(lines).lower()

    # Check for presence of all required %chelpg parameters
    required = ['%chelpg', 'grid 0.3', 'rmax 2.8', 'vdwradii cosmo']
    return all(r in embedded for r in required)


def _h_atom_inp(job_dir: Path, nprocs: int, method: str) -> Path:
    inp = job_dir / "H_atom_sp.inp"
    inp.write_text(
        f"! {method} SP TightSCF\n"
        f"%pal nprocs {nprocs} end\n"
        f"* xyz 0 2\n"
        f"H  0.0  0.0  0.0\n"
        f"*\n"
    )
    return inp

# ──────────────────────────────────────────────────────────────────────────────
# Per-microstate ensemble calculation
# ──────────────────────────────────────────────────────────────────────────────

def run_ensemble(
        ms:           Microstate,
        work_dir:     Path,
        orca_binary:  str,
        nprocs:       int,
        nconf:        int,
        ewindow:      float,
        method:       str,
        do_tautomers: bool,
        temperature:  float,
) -> None:
    """
    Three-stage calculation for one microstate.

    Stage A — GFN2-xTB/ALPB GOAT: conformer + (optional) tautomer search.
              Produces a ranked ensemble of structures.

    Stage B — GFN2-xTB/ALPB OPT+FREQ for each kept structure.
              Produces G_RRHO (thermal correction) and E_xTB for each.

    Stage C — DFT/CPCM SP for each kept structure (on the OPT geometry).
              Produces E_DFT for each.

    Composite free energy (Eq. 2):
              G_c = E_DFT,c + (G_RRHO,c - E_xTB,c)

    Ensemble effective free energy (Eq. 1):
              G_eff = -kT ln Σ exp(-G_c / kT)

    Populates ms.G_eff, ms.structures, ms.n_struct_found, ms.n_struct_used.
    """
    solvent_xtb = "ALPB(Water)"
    solvent_dft = "CPCM(Water)"
    kT          = KB_EV * temperature / EH_TO_EV

    ms_dir = work_dir / ms.name
    ms_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage A: GOAT ─────────────────────────────────────────────────────────
    logging.info("[%s] Stage A — GOAT conformer search…", ms.name)
    goat_dir = ms_dir / "goat"
    goat_dir.mkdir(exist_ok=True)
    goat_inp_path = _goat_inp(ms, goat_dir, nprocs, solvent_xtb, do_tautomers)
    _run_orca(goat_inp_path, orca_binary)

    xyz_blocks = _parse_goat_ensemble_xyz(goat_dir)
    ms.n_struct_found = len(xyz_blocks) or 1

    if not xyz_blocks:
        logging.warning("[%s] No ensemble found — using input geometry.", ms.name)
        xyz_blocks = [ms.xyz_block]

    # Keep at most nconf structures (GOAT writes in energy order)
    kept_xyz = xyz_blocks[:nconf]
    ms.n_struct_used = len(kept_xyz)
    logging.info("[%s] GOAT: %d structures found, %d kept (max=%d).",
                 ms.name, ms.n_struct_found, ms.n_struct_used, nconf)

    # ── Stages B + C: xTB FREQ + DFT SP ──────────────────────────────────────
    structures: list[Structure] = []
    n_skipped = 0
    for idx, xyz in enumerate(kept_xyz):
        label    = f"{ms.name}_s{idx:03d}"
        sdir     = ms_dir / f"s{idx:03d}"
        sdir.mkdir(exist_ok=True)

        # Stage B: GFN2-xTB OPT+FREQ
        logging.info("[%s  s%03d] Stage B — GFN2-xTB OPT+FREQ…", ms.name, idx)
        f_inp = _freq_inp(ms, xyz, sdir, label, nprocs, solvent_xtb, robust=False)
        _run_orca(f_inp, orca_binary)
        f_out = sdir / f"{label}_freq.out"
        try:
            G_xtb, E_xtb = _parse_gibbs_and_sp(f_out)
        except (SCFDivergenceError, MissingThermoError) as e:
            # ── Retry with robust SCF settings ────────────────────────────────
            # Both SCF divergence and missing thermochemistry indicate the
            # standard settings failed.  Retry once with:
            #   - Electronic temperature 3000 K (prevents orbital crossings)
            #   - Broyden damping 0.7 (prevents overshoot)
            #   - SlowConv strategy + 500 SCF iterations
            # The retry uses a subdirectory to preserve the failed output.
            logging.warning("[%s  s%03d] Initial OPT+FREQ failed (%s). "
                            "Retrying with robust SCF settings…",
                            ms.name, idx, type(e).__name__)
            retry_dir = sdir / "retry"
            retry_dir.mkdir(exist_ok=True)
            retry_label = f"{label}_retry"
            # Use the optimised geometry from the failed run if available,
            # otherwise fall back to the GOAT geometry.
            retry_xyz = _read_optimised_xyz(f_out) or xyz
            r_inp = _freq_inp(ms, retry_xyz, retry_dir, retry_label,
                              nprocs, solvent_xtb, robust=True)
            try:
                _run_orca(r_inp, orca_binary)
                r_out = retry_dir / f"{retry_label}_freq.out"
                G_xtb, E_xtb = _parse_gibbs_and_sp(r_out)
                # Use the retry output for the optimised geometry
                f_out = r_out
                logging.info("[%s  s%03d] Robust retry succeeded: "
                             "G_xTB=%.8f  ΔG_RRHO=%.4f kcal/mol",
                             ms.name, idx, G_xtb, (G_xtb - E_xtb) * EH_TO_KCAL)
            except (SCFDivergenceError, MissingThermoError, RuntimeError) as e2:
                # ── Second retry: VerySlowConv + etemp 5000 ───────────────────
                # If SlowConv + etemp 3000 still fails, escalate to VerySlowConv
                # and a higher electronic temperature (5000 K) which broadens
                # fractional occupation enough to smooth pathological orbital
                # crossings in rigid aromatic cations (e.g. 3-chloroanilinium).
                logging.warning("[%s  s%03d] Robust retry failed (%s). "
                                "Attempting VerySlowConv recovery…",
                                ms.name, idx, e2)
                retry2_dir = sdir / "retry2"
                retry2_dir.mkdir(exist_ok=True)
                retry2_label = f"{label}_retry2"
                retry2_xyz = _read_optimised_xyz(retry_dir / f"{retry_label}_freq.out") or retry_xyz
                r2_inp = _freq_inp(ms, retry2_xyz, retry2_dir, retry2_label,
                                   nprocs, solvent_xtb, robust=True,
                                   very_robust=True)
                try:
                    _run_orca(r2_inp, orca_binary)
                    r2_out = retry2_dir / f"{retry2_label}_freq.out"
                    G_xtb, E_xtb = _parse_gibbs_and_sp(r2_out)
                    f_out = r2_out
                    logging.info("[%s  s%03d] VerySlowConv recovery succeeded: "
                                 "G_xTB=%.8f", ms.name, idx, G_xtb)
                except (SCFDivergenceError, MissingThermoError, RuntimeError) as e3:
                    logging.warning("[%s  s%03d] All retry levels failed (%s). "
                                    "Skipping conformer.",
                                    ms.name, idx, e3)
                    n_skipped += 1
                    continue
        logging.info("[%s  s%03d] G_xTB=%.8f  E_xTB=%.8f  ΔG_RRHO=%.4f kcal/mol",
                     ms.name, idx, G_xtb, E_xtb, (G_xtb - E_xtb) * EH_TO_KCAL)

        # Use optimised geometry for DFT SP
        opt_xyz = _read_optimised_xyz(f_out) or xyz

        # Stage C: DFT SP
        logging.info("[%s  s%03d] Stage C — %s SP…", ms.name, idx, method)
        d_inp = _dft_sp_inp(ms, opt_xyz, sdir, label, nprocs, method, solvent_dft)
        _run_orca(d_inp, orca_binary)
        d_out = sdir / f"{label}_dft.out"
        try:
            E_dft = _parse_sp_energy(d_out)
        except (ValueError, RuntimeError) as e_dft:
            # DFT SP convergence failure — retry with SlowConv
            logging.warning("[%s  s%03d] DFT SP failed (%s). "
                            "Retrying with SlowConv…", ms.name, idx, e_dft)
            dr_inp = _dft_sp_inp(ms, opt_xyz, sdir, label + "_dft_retry",
                                 nprocs, method, solvent_dft, slow_conv=True)
            _run_orca(dr_inp, orca_binary)
            dr_out = sdir / f"{label}_dft_retry_dft.out"
            try:
                E_dft = _parse_sp_energy(dr_out)
                d_out = dr_out
                logging.info("[%s  s%03d] DFT SP retry succeeded.", ms.name, idx)
            except (ValueError, RuntimeError) as e_dft2:
                logging.warning("[%s  s%03d] DFT SP retry also failed (%s). "
                                "Skipping conformer.", ms.name, idx, e_dft2)
                n_skipped += 1
                continue
        G_comp  = E_dft + (G_xtb - E_xtb)
        logging.info("[%s  s%03d] E_DFT=%.8f  G_composite=%.8f Eh",
                     ms.name, idx, E_dft, G_comp)

        structures.append(Structure(
            label       = label,
            xyz_block   = opt_xyz,
            G_xtb_aq    = G_xtb,
            E_xtb_aq    = E_xtb,
            E_dft_aq    = E_dft,
            G_composite = G_comp,
        ))

    if n_skipped:
        logging.warning("[%s] %d/%d conformers skipped due to SCF divergence.",
                        ms.name, n_skipped, len(kept_xyz))

    # Apply energy window filter on composite G
    if not structures:
        raise RuntimeError(
            f"[{ms.name}] All {len(kept_xyz)} conformers had SCF divergence "
            f"({n_skipped} skipped). Cannot compute G_eff. "
            f"Consider: (1) re-running GOAT with fewer conformers, "
            f"(2) providing a manually optimised starting geometry, or "
            f"(3) checking the .msf charge/multiplicity for this microstate."
        )

    G_min = min(s.G_composite for s in structures)
    kept  = [s for s in structures
             if (s.G_composite - G_min) * EH_TO_KCAL <= ewindow]
    if not kept:
        kept = [structures[0]]

    # Boltzmann average (Eq. 1)
    G_arr    = np.array([s.G_composite for s in kept])
    G_eff    = float(-kT * np.log(np.sum(np.exp(-(G_arr - G_min) / kT)))) + G_min
    dG_ens   = (G_eff - G_min) * EH_TO_KCAL

    ms.G_eff       = G_eff
    ms.structures  = kept
    ms.n_struct_used = len(kept)
    logging.info("[%s] G_eff = %.8f Eh  (ensemble correction = %.4f kcal/mol, %d structures)",
                 ms.name, G_eff, dG_ens, len(kept))

# ──────────────────────────────────────────────────────────────────────────────
# H-atom SP → G*(H+,aq)
# ──────────────────────────────────────────────────────────────────────────────

def compute_G_proton(
        work_dir:    Path,
        orca_binary: str,
        nprocs:      int,
        method:      str,
        temperature: float,
) -> float:
    """
    Absolute cycle for G*(H+,aq) (Eq. 3).
    Runs one DFT SP on a neutral H atom in the gas phase.
    """
    h_dir = work_dir / "_H_atom_sp"
    h_dir.mkdir(parents=True, exist_ok=True)
    h_inp = _h_atom_inp(h_dir, nprocs, method)
    logging.info("Running H-atom SP (%s, gas phase)…", method)
    _run_orca(h_inp, orca_binary)
    h_out    = h_dir / "H_atom_sp.out"
    G_H_gas  = _parse_sp_energy(h_out)
    G_proton = G_H_gas + IE_HYDROGEN_EH + DG_SOLV_PROTON_EH
    logging.info(
        "G*(H+,aq) = %.6f + %.6f + %.6f = %.6f Eh (%.2f kcal/mol)",
        G_H_gas, IE_HYDROGEN_EH, DG_SOLV_PROTON_EH,
        G_proton, G_proton * EH_TO_KCAL,
    )
    return G_proton

# ──────────────────────────────────────────────────────────────────────────────
# Population and pKa
# ──────────────────────────────────────────────────────────────────────────────

def compute_populations(
        microstates: list[Microstate],
        G_proton:    "float | list[float]",
        pH_array:    np.ndarray,
        temperature: float = TEMPERATURE,
) -> np.ndarray:
    """
    Boltzmann populations vs pH using per-step G*(H+).

    G_proton may be a single float (same G*(H+) for all steps) or a list
    of floats (one per ionisation step, step-1-first, i.e. the step that
    removes the proton from the most protonated form first).

    Returns array of shape (N_microstates, N_pH).
    """
    kT    = KB_EV * temperature / EH_TO_EV
    G_eff = np.array([ms.G_eff for ms in microstates])
    n_H   = np.array([ms.n_protons for ms in microstates], dtype=float)
    N     = int(n_H.max())

    if isinstance(G_proton, (int, float, np.floating)):
        G_list = [float(G_proton)] * N
    else:
        G_list = list(G_proton)
        if len(G_list) < N:
            G_list = G_list + [G_list[-1]] * (N - len(G_list))
        G_list = G_list[:N]

    # cumulative[n] = sum of the LAST n G*(H+) steps
    cumulative = [0.0] * (N + 1)
    for n in range(1, N + 1):
        cumulative[n] = sum(G_list[N - n:])

    G_corr  = np.array([ms.G_eff - cumulative[ms.n_protons] for ms in microstates])
    ref_idx = next(i for i, ms in enumerate(microstates) if ms.n_protons == 0)
    G_ref   = G_corr[ref_idx]

    beta_dG0 = (G_corr - G_ref) / kT
    beta_dG  = beta_dG0[:, None] + n_H[:, None] * LN10 * pH_array[None, :]

    log_w = -beta_dG
    log_w -= log_w.max(axis=0, keepdims=True)
    w = np.exp(log_w)
    return w / w.sum(axis=0, keepdims=True)



def compute_macropka(
        microstates:  list[Microstate],
        pH_array:     np.ndarray,
        populations:  np.ndarray,
) -> list[float]:
    """
    Macroscopic pKa from population crossings.
    pKa_k is where the k-th proton is 50% ionised (population crossing).
    """
    sorted_ms = sorted(microstates, key=lambda m: m.n_protons, reverse=True)
    n_steps   = len(microstates) - 1
    pkas      = []
    for k in range(n_steps):
        thresh  = sorted_ms[k + 1].n_protons
        idx_hi  = [i for i, ms in enumerate(microstates) if ms.n_protons > thresh]
        idx_lo  = [i for i, ms in enumerate(microstates) if ms.n_protons <= thresh]
        P_hi    = populations[idx_hi, :].sum(axis=0)
        P_lo    = populations[idx_lo, :].sum(axis=0)
        diff    = P_hi - P_lo
        cross   = np.where(np.diff(np.sign(diff)))[0]
        if len(cross):
            i0     = cross[0]
            d0, d1 = diff[i0], diff[i0 + 1]
            p0, p1 = pH_array[i0], pH_array[i0 + 1]
            pkas.append(float(p0 - d0 * (p1 - p0) / (d1 - d0)))
        else:
            pkas.append(float('nan'))
    return pkas

# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def plot_populations(
        microstates:  list[Microstate],
        pH_array:     np.ndarray,
        populations:  np.ndarray,
        pka_pred:     list[float],
        mol_name:     str,
        method:       str,
        out_path:     Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = cm.tab10(np.linspace(0, 0.9, len(microstates)))

    for ax, logscale in [(ax1, False), (ax2, True)]:
        for i, ms in enumerate(microstates):
            label = f"{ms.name}  (z={ms.charge:+d}, nH={ms.n_protons})"
            pop   = populations[i]
            if logscale:
                ax.semilogy(pH_array, np.clip(pop, 1e-9, 1),
                            lw=2, color=colors[i], label=label)
            else:
                ax.plot(pH_array, pop, lw=2, color=colors[i], label=label)

        for pka in pka_pred:
            if not np.isnan(pka):
                ax.axvline(pka, color='grey', ls='--', lw=1.0, alpha=0.7)
                ax.text(pka + 0.1, 0.93, f"pKa={pka:.2f}",
                        transform=ax.get_xaxis_transform(),
                        va='top', fontsize=7.5, color='dimgrey')

        ax.set_xlim(pH_array[0], pH_array[-1])
        ax.set_xlabel('pH', fontsize=12)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper right' if not logscale else 'lower right',
                  framealpha=0.9)

    ax1.set_ylim(-0.02, 1.05)
    ax1.set_ylabel('Fractional population', fontsize=12)
    ax1.set_title('Linear scale', fontsize=11)
    ax2.set_ylabel('Fractional population (log)', fontsize=12)
    ax2.set_title('Log scale', fontsize=11)
    fig.suptitle(
        f"{mol_name}\n"
        f"{method}/CPCM(Water)  +  GFN2-xTB/ALPB RRHO  —  "
        f"pKa: {[f'{p:.2f}' if not np.isnan(p) else 'nan' for p in pka_pred]}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    logging.info("Plot saved → %s", out_path)
    plt.close(fig)


def save_results(
        mol_name:    str,
        microstates: list[Microstate],
        G_proton:    float,
        pH_array:    np.ndarray,
        populations: np.ndarray,
        pka_pred:    list[float],
        method:      str,
        out_path:    Path,
) -> None:
    data = {
        'molecule':        mol_name,
        'method':          method,
        'G_proton_aq_Eh':  G_proton if isinstance(G_proton, list) else [G_proton],
        'pKa_predicted':   pka_pred,
        'microstates': [
            {
                'name':           ms.name,
                'charge':         ms.charge,
                'n_protons':      ms.n_protons,
                'G_eff_Eh':       ms.G_eff,
                'n_struct_found': ms.n_struct_found,
                'n_struct_used':  ms.n_struct_used,
                'structures': [
                    {
                        'label':       s.label,
                        'G_xtb_aq':   s.G_xtb_aq,
                        'E_xtb_aq':   s.E_xtb_aq,
                        'E_dft_aq':   s.E_dft_aq,
                        'G_composite':s.G_composite,
                    }
                    for s in ms.structures
                ],
            }
            for ms in microstates
        ],
        'pH_array':    pH_array.tolist(),
        'populations': populations.tolist(),
    }
    out_path.write_text(json.dumps(data, indent=2))
    logging.info("Results saved → %s", out_path)

# ──────────────────────────────────────────────────────────────────────────────
# Dry-run mock
# ──────────────────────────────────────────────────────────────────────────────

def _mock_run(microstates: list[Microstate], G_proton: float,
              temperature: float) -> None:
    """
    Inject plausible synthetic G_eff values for pipeline testing.
    Uses spaced-out pKa values (4, 7, 10, ...) as mock targets.
    """
    RT_ln10   = KB_EV * temperature / EH_TO_EV * LN10
    ms_sorted = sorted(microstates, key=lambda m: m.n_protons, reverse=True)
    n_steps   = len(ms_sorted) - 1
    mock_pkas = [4.0 + k * 3.0 for k in range(n_steps)]

    G_ref_val = -55.0
    ms_sorted[-1].G_eff          = G_ref_val
    ms_sorted[-1].n_struct_found = 1
    ms_sorted[-1].n_struct_used  = 1

    current = G_ref_val
    for k, (ms, pka) in enumerate(zip(reversed(ms_sorted[:-1]),
                                       reversed(mock_pkas))):
        # From: pKa = (G(base) + G*(H+) - G(acid)) / (RT ln10)
        # => G(acid) = G*(H+) + G(base) - pKa * RT ln10
        G_acid               = G_proton + current - pka * RT_ln10
        ms.G_eff             = G_acid
        ms.n_struct_found    = 1
        ms.n_struct_used     = 1
        current              = G_acid

# ──────────────────────────────────────────────────────────────────────────────
# G_eff cache  (used by pka_gpr_calibrate without needing pka_calibrate)
# ──────────────────────────────────────────────────────────────────────────────

def _geff_cache_path(msf_path: Path, out_dir: Path) -> Path:
    return out_dir / f"{msf_path.stem}_geff.json"


def load_cached_G_eff(msf_path: Path, out_dir: Path) -> Optional[dict[str, float]]:
    """
    Load G_eff values from the compound's JSON cache.
    Returns dict[ms_name → G_eff_Eh] or None if cache missing / corrupt.
    """
    cache = _geff_cache_path(msf_path, out_dir)
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
    """Persist G_eff values for all microstates to a JSON sidecar file."""
    cache = _geff_cache_path(msf_path, out_dir)
    data  = {
        "compound":    msf_path.stem,
        "method":      method,
        "microstates": [
            {
                "name":           ms.name,
                "charge":         ms.charge,
                "n_protons":      ms.n_protons,
                "G_eff_Eh":       ms.G_eff,
                "n_struct_found": ms.n_struct_found,
                "n_struct_used":  ms.n_struct_used,
            }
            for ms in microstates
        ],
    }
    cache.write_text(json.dumps(data, indent=2))
    logging.info("  [%s] G_eff cached → %s", msf_path.stem, cache.name)


def run_orca_for_compound(
        msf_path:     Path,
        out_dir:      Path,
        orca_binary:  str  = "orca",
        nprocs:       int  = 4,
        nconf:        int  = 10,
        ewindow:      float = 3.0,
        method:       str  = "r2SCAN-3c",
        do_tautomers: bool = True,
        temperature:  float = 298.15,
        dry_run:      bool = False,
) -> list[Microstate]:
    """
    Run (or load) GOAT + xTB FREQ + DFT SP for all microstates in an MSF file.

    Checks the G_eff cache first; only runs ORCA when the cache is absent or
    incomplete.  Saves the cache after a successful ORCA run.

    Returns the list of Microstate objects with G_eff populated (or None when
    ORCA failed for a microstate).  The caller is responsible for any further
    analysis (descriptors, LFER, GPR).
    """
    method_kw = {
        "r2scan3c": "r2SCAN-3c", "r2scan-3c": "r2SCAN-3c",
        "pbe3c":    "PBE-3c",    "pbe-3c":    "PBE-3c",
    }.get(method.lower(), method)

    compound    = msf_path.stem
    microstates = parse_msf(msf_path)
    work_dir    = out_dir / f"work_{compound}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load cache if complete
    cached = load_cached_G_eff(msf_path, out_dir)
    if cached is not None:
        missing = [ms.name for ms in microstates if ms.name not in cached]
        if missing:
            logging.warning("[%s] Cache missing entries %s — recomputing.",
                            compound, missing)
            cached = None
        else:
            for ms in microstates:
                ms.G_eff         = cached[ms.name]
                ms.n_struct_used = 1
            return microstates

    # Cache absent or incomplete — run ORCA
    if dry_run:
        _mock_run(microstates, -0.447472, temperature)
    else:
        logging.info("[%s] Running ORCA ensemble calculations…", compound)
        for ms in microstates:
            run_ensemble(
                ms=ms, work_dir=work_dir, orca_binary=orca_binary,
                nprocs=nprocs, nconf=nconf, ewindow=ewindow,
                method=method_kw, do_tautomers=do_tautomers,
                temperature=temperature,
            )

    save_G_eff_cache(msf_path, microstates, method_kw, out_dir)
    return microstates

# ──────────────────────────────────────────────────────────────────────────────
# .msf template generator
# ──────────────────────────────────────────────────────────────────────────────

def make_msf_template(xyz_paths: list[Path], out_path: Path) -> None:
    """Generate a .msf template from a list of .xyz files."""
    blocks = []
    for i, p in enumerate(sorted(xyz_paths)):
        raw   = p.read_text()
        lines = raw.splitlines()
        try:
            n_at = int(lines[0].strip())
            body = '\n'.join(lines[2: 2 + n_at])
        except (ValueError, IndexError):
            body = raw.strip()

        blocks.append(textwrap.dedent(f"""\
            [microstate]
            name         = {p.stem}
            charge       = 0        # EDIT: formal charge of this protonation state
            multiplicity = 1        # EDIT: spin multiplicity (1 = closed-shell)
            n_protons    = {i}       # EDIT: 0 = fully deprotonated reference
                                    #        1, 2, ... = number of titratable H

            xyz =
            {body}
        """))

    header = textwrap.dedent(f"""\
        # .msf microstate file — generated by pka_calc.py --make-msf
        # Sources: {', '.join(p.name for p in xyz_paths)}
        #
        # Edit charge, multiplicity, n_protons for each block.
        # Rules:
        #   - Exactly ONE block must have n_protons = 0 (fully deprotonated).
        #   - n_protons must be consecutive integers: 0, 1, 2, ...
        #   - No experimental pKa values are needed.

    """)
    out_path.write_text(header + '\n'.join(blocks))
    print(f"Template written → {out_path}")
    print("Edit charge / multiplicity / n_protons before running.")

# ──────────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────────

def load_lfer_params(lfer_json: Path) -> dict:
    """
    Load LFER parameters from lfer_params.json (written by pka_calibrate.py).
    Returns a dict keyed by 'functional_group_q<acid_charge>',
    e.g. {'carboxylate_q+0': {...}, 'ammonium_q+0': {...}, 'ammonium_q+1': {...}}.
    """
    data = json.loads(lfer_json.read_text())
    return data.get("functional_groups", {})


def _lfer_key(fg: str, acid_charge: int) -> str:
    """Canonical key matching pka_calibrate.py: e.g. 'ammonium_q+0'."""
    return f"{fg}_q{acid_charge:+d}"


def check_lfer_coverage(lfer_params: dict,
                        microstates: list[Microstate]) -> None:
    """
    Verify that every ionisation step in the target has both a
    functional_group annotation and a matching LFER key in lfer_params.
    Logs warnings for any gaps.
    """
    sorted_ms = sorted(microstates, key=lambda m: m.n_protons, reverse=True)
    ok = True
    for ms_acid, ms_base in zip(sorted_ms[:-1], sorted_ms[1:]):
        fg = ms_base.functional_group
        if fg is None:
            logging.warning(
                "  Step (%s → %s): no functional_group on '%s'. "
                "Add 'functional_group = <group>' to the .msf file.",
                ms_acid.name, ms_base.name, ms_base.name,
            )
            ok = False
            continue
        key = _lfer_key(fg, ms_acid.charge)
        if key in lfer_params:
            logging.info("  Step (%s → %s): key='%s' ✓",
                         ms_acid.name, ms_base.name, key)
        else:
            # Try same fg, any charge — find nearest
            candidates = [k for k in lfer_params
                          if lfer_params[k].get("functional_group", "") == fg]
            if candidates:
                logging.warning(
                    "  Step (%s → %s): key='%s' not found. "
                    "Available for %s: %s  — will use nearest charge.",
                    ms_acid.name, ms_base.name, key, fg, candidates,
                )
            else:
                logging.warning(
                    "  Step (%s → %s): functional_group='%s' has no LFER at all. "
                    "Available keys: %s",
                    ms_acid.name, ms_base.name, fg, list(lfer_params),
                )
                ok = False
    if ok:
        logging.info("LFER coverage check: all steps covered ✓")


def apply_lfer_per_step(
        microstates:  list[Microstate],
        lfer_params:  dict,
        temperature:  float = TEMPERATURE,
) -> list[float]:
    """
    Apply LFER per step.  For each step looks up the exact key
    'functional_group_q<acid_charge>', e.g. 'ammonium_q+0'.

    Fallback: if the exact key is absent, uses the entry for the same
    functional group with the nearest acid charge and logs a warning.

    Returns one G*(H+)_eff per step, step-1-first.
    """
    RT_ln10   = KB_EV * temperature / EH_TO_EV * LN10
    sorted_ms = sorted(microstates, key=lambda m: m.n_protons, reverse=True)
    result    = []

    for ms_acid, ms_base in zip(sorted_ms[:-1], sorted_ms[1:]):
        dG    = ms_base.G_eff - ms_acid.G_eff
        pka_c = dG / RT_ln10
        fg    = ms_base.functional_group
        q     = ms_acid.charge
        key   = _lfer_key(fg, q) if fg else None

        if key and key in lfer_params:
            params = lfer_params[key]
            used   = key
        else:
            # Find nearest available key for the same functional group
            candidates = {k: v for k, v in lfer_params.items()
                          if v.get("functional_group", "") == fg}
            if candidates:
                # Pick the candidate whose acid_charge is closest
                used   = min(candidates,
                             key=lambda k: abs(lfer_params[k].get("acid_charge", 0) - q))
                params = lfer_params[used]
                logging.warning(
                    "  Step (%s → %s, fg=%s, q=%+d): exact key '%s' not found. "
                    "Using '%s' (nearest charge). Accuracy may be reduced.",
                    ms_acid.name, ms_base.name, fg, q, key, used,
                )
            elif lfer_params:
                used   = next(iter(lfer_params))
                params = lfer_params[used]
                logging.warning(
                    "  Step (%s → %s): fg='%s' has no LFER at all. "
                    "Falling back to '%s'. Accuracy significantly reduced.",
                    ms_acid.name, ms_base.name, fg, used,
                )
            else:
                raise ValueError("lfer_params is empty — run pka_calibrate.py first.")

        slope    = params["slope"]
        intercept= params["intercept"]
        pka_pred = slope * pka_c + intercept
        G_H_eff  = pka_pred * RT_ln10 - ms_base.G_eff + ms_acid.G_eff
        result.append(G_H_eff)

        logging.info(
            "  LFER (%s → %s)  key=%-22s  "
            "ΔG_DFT=%+.2f kcal/mol  slope=%.4f  intercept=%.3f  pKa_pred=%.3f",
            ms_acid.name, ms_base.name, used,
            dG * EH_TO_KCAL, slope, intercept, pka_pred,
        )

    return result


def run_calculation(
        msf_path:     Path,
        ref_msf:      Optional[Path],
        lfer_json:    Optional[Path],
        orca_binary:  str,
        nprocs:       int,
        nconf:        int,
        ewindow:      float,
        method:       str,
        do_tautomers: bool,
        ph_min:       float,
        ph_max:       float,
        ph_step:      float,
        temperature:  float,
        dry_run:      bool,
        out_dir:      Path,
) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)
    mol_name = msf_path.stem
    work_dir = out_dir / f"work_{mol_name}"
    work_dir.mkdir(parents=True, exist_ok=True)

    method_kw = {
        'r2scan3c':  'r2SCAN-3c',
        'r2scan-3c': 'r2SCAN-3c',
        'pbe3c':     'PBE-3c',
        'pbe-3c':    'PBE-3c',
    }.get(method.lower(), method)

    # Determine proton reference mode
    if lfer_json is not None:
        ref_mode = f"LFER ({lfer_json.name})"
    elif ref_msf is not None:
        ref_mode = f"isodesmic ({ref_msf.stem})"
    else:
        ref_mode = "absolute cycle (use --ref or --lfer for better accuracy)"

    logging.info("=" * 62)
    logging.info("pka_calc.py — %s", mol_name)
    logging.info("  DFT method:  %s / CPCM(Water)", method_kw)
    logging.info("  xTB level:   GFN2-xTB / ALPB(Water)  (RRHO + search)")
    logging.info("  Tautomers:   %s", "via separate .msf microstates (GOAT has no built-in tautomer search in ORCA 6.x)" if do_tautomers else "no")
    logging.info("  Max struct:  %d per microstate (window %.1f kcal/mol)", nconf, ewindow)
    logging.info("  Ref mode:    %s", ref_mode)
    logging.info("=" * 62)

    microstates = parse_msf(msf_path)
    logging.info("Microstates: %s", [f"{ms.name}(nH={ms.n_protons},z={ms.charge:+d})"
                                      for ms in microstates])

    # ── Load LFER parameters if provided ─────────────────────────────────────
    lfer_params: Optional[dict] = None
    if lfer_json is not None:
        lfer_params = load_lfer_params(lfer_json)
        check_lfer_coverage(lfer_params, microstates)

    ref_microstates: Optional[list[Microstate]] = None
    if ref_msf is not None and lfer_params is None:
        ref_microstates = parse_msf(ref_msf)
        logging.info("Reference:   %s", [f"{ms.name}(nH={ms.n_protons})"
                                          for ms in ref_microstates])

    # ── Stage 0: ensemble calculations ───────────────────────────────────────
    all_ms_to_run = []
    if ref_microstates is not None:
        all_ms_to_run.append((ref_microstates, out_dir / f"work_{ref_msf.stem}"))
    all_ms_to_run.append((microstates, work_dir))

    if not dry_run:
        for ms_list, w_dir in all_ms_to_run:
            w_dir.mkdir(parents=True, exist_ok=True)
            for ms in ms_list:
                run_ensemble(
                    ms           = ms,
                    work_dir     = w_dir,
                    orca_binary  = orca_binary,
                    nprocs       = nprocs,
                    nconf        = nconf,
                    ewindow      = ewindow,
                    method       = method_kw,
                    do_tautomers = do_tautomers,
                    temperature  = temperature,
                )
    else:
        # Dry-run: inject approximate G*(H+) then mock G_eff
        # Dry-run: inject plausible G_proton and mock G_eff
        G_proton_approx = -0.447472  # approximate r2SCAN-3c isodesmic from AcOH
        for ms_list, _ in all_ms_to_run:
            _mock_run(ms_list, G_proton_approx, temperature)

    # ── Stage 1: derive G*(H+,aq) ────────────────────────────────────────────
    if lfer_params is not None:
        # LFER mode: slope + intercept per charge transition
        logging.info("")
        logging.info("Applying LFER parameters per charge transition…")
        G_proton = apply_lfer_per_step(microstates, lfer_params, temperature)
        logging.info("  G*(H+)_eff per step: %s Eh",
                     [f"{g:.5f}" for g in G_proton])

    elif ref_microstates is not None:
        logging.info("")
        logging.info("Deriving per-step G*(H+,aq) from '%s'…", ref_msf.stem)
        G_H_by_charge = derive_G_proton_per_step(ref_microstates, temperature)
        logging.info("  Reference G*(H+) by acid charge: %s",
                     {q: f"{v*EH_TO_KCAL:.2f} kcal/mol"
                      for q, v in sorted(G_H_by_charge.items())})

        logging.info("Mapping to target steps by acid charge…")
        G_proton_list = map_G_proton_to_target(G_H_by_charge, microstates)
        logging.info("  Target G*(H+) per step: %s Eh",
                     [f"{g:.5f}" for g in G_proton_list])
        G_proton = G_proton_list   # list, one per step
    else:
        # Absolute cycle fallback — ~13 pKa unit error at r2SCAN-3c/CPCM
        logging.warning(
            "No reference compound provided.  Using absolute cycle "
            "(G*(H+) from H-atom SP).  Expect ~13 pKa unit systematic error "
            "at r2SCAN-3c/CPCM level.  Provide --ref for accurate results."
        )
        if not dry_run:
            G_proton = compute_G_proton(
                work_dir, orca_binary, nprocs, method_kw, temperature)
        else:
            G_proton = -0.418217   # approximate absolute value (dry-run only)
            logging.warning("Dry-run: using approximate absolute G*(H+) = %.6f Eh", G_proton)

    # ── Stage 2: populations and pKa ─────────────────────────────────────────
    logging.info("")
    logging.info("G_eff summary (target):")
    for ms in sorted(microstates, key=lambda m: m.n_protons, reverse=True):
        logging.info("  %-22s  nH=%d  z=%+d  G_eff=%.8f Eh  (%d structures)",
                     ms.name, ms.n_protons, ms.charge,
                     ms.G_eff, ms.n_struct_used)

    pH_array    = np.arange(ph_min, ph_max + ph_step / 2, ph_step)
    populations = compute_populations(microstates, G_proton, pH_array, temperature)
    pka_pred    = compute_macropka(microstates, pH_array, populations)

    logging.info("")
    logging.info("=" * 62)
    logging.info("RESULTS — %s", mol_name)
    logging.info("  %s / CPCM + GFN2-xTB RRHO", method_kw)
    logging.info("  G*(H+,aq) = %s Eh  (isodesmic from %s)",
                 [f"{g:.5f}" for g in G_proton] if isinstance(G_proton, list)
                 else f"{G_proton:.6f}",
                 ref_msf.stem if ref_msf else "absolute cycle")
    for k, pka in enumerate(pka_pred):
        logging.info("  pKa%d = %s", k + 1,
                     f"{pka:.3f}" if not np.isnan(pka) else "nan")
    logging.info("=" * 62)

    stem = mol_name
    plot_populations(microstates, pH_array, populations, pka_pred,
                     mol_name, method_kw,
                     out_dir / f"{stem}_populations.png")
    save_results(mol_name, microstates, G_proton,
                 pH_array, populations, pka_pred,
                 method_kw, out_dir / f"{stem}_results.json")

    return {
        'microstates': microstates,
        'pH_array':    pH_array,
        'populations': populations,
        'pka_pred':    pka_pred,
        'G_proton':    G_proton,
    }

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%H:%M:%S',
    )

    p = argparse.ArgumentParser(
        prog='pka_calc.py',
        description='Macroscopic pKa from conformational/tautomeric ensembles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            # Generate .msf template from xyz files:
            python pka_calc.py --make-msf HA.xyz A.xyz --out molecule.msf

            # Full calculation (r2SCAN-3c, tautomers on):
            python pka_calc.py molecule.msf \\
                --orca /apps/software/orca/orca-6.1.0/orca --nprocs 12

            # Faster with PBE-3c:
            python pka_calc.py molecule.msf \\
                --orca /apps/software/orca/orca-6.1.0/orca --nprocs 12 --method pbe3c

            # No tautomer search:
            python pka_calc.py molecule.msf --orca ... --no-tautomers

            # Dry-run (no ORCA, test pipeline):
            python pka_calc.py molecule.msf --dry-run
        """),
    )

    p.add_argument('--make-msf', nargs='+', metavar='XYZ',
                   help='Generate .msf template from xyz files, then exit.')
    p.add_argument('--out', default=None, metavar='FILE',
                   help='Output path for --make-msf.')

    p.add_argument('msf', nargs='?', metavar='MOLECULE.msf')

    p.add_argument('--ref', default=None, metavar='REFERENCE.msf',
                   help=(
                       '.msf file for the isodesmic reference compound. '
                       'Must have pka_step = <experimental_pKa> on its base '
                       'microstate(s).  Mutually exclusive with --lfer.'
                   ))
    p.add_argument('--lfer', default=None, metavar='LFER_PARAMS.json',
                   help=(
                       'lfer_params.json from pka_calibrate.py.  '
                       'Applies fitted slope+intercept per charge transition '
                       'for maximum accuracy.  Preferred over --ref when '
                       'a calibration has been run.  Mutually exclusive with --ref.'
                   ))

    p.add_argument('--orca',    default='orca', metavar='PATH')
    p.add_argument('--nprocs',  type=int, default=4,    metavar='N')
    p.add_argument('--nconf',   type=int, default=10,   metavar='N',
                   help='Max conformers/tautomers per microstate. (default: 10)')
    p.add_argument('--ewindow', type=float, default=3.0, metavar='KCAL',
                   help='Ensemble energy window in kcal/mol. (default: 3.0)')
    p.add_argument('--method',  default='r2scan3c', metavar='METHOD',
                   help='DFT SP method: r2scan3c (default) or pbe3c.')
    p.add_argument('--no-tautomers', action='store_true',
                   help='Disable tautomer search in GOAT.')
    p.add_argument('--temp',    type=float, default=298.15, metavar='K')
    p.add_argument('--ph-min',  type=float, default=0.0)
    p.add_argument('--ph-max',  type=float, default=14.0)
    p.add_argument('--ph-step', type=float, default=0.05)
    p.add_argument('--outdir',  default='pka_results', metavar='DIR')
    p.add_argument('--dry-run', action='store_true')

    args = p.parse_args()

    if args.make_msf:
        out = Path(args.out) if args.out else Path(args.make_msf[0]).with_suffix('.msf')
        make_msf_template([Path(x) for x in args.make_msf], out)
        return

    if not args.msf:
        p.error('Provide MOLECULE.msf, or use --make-msf.')

    if args.ref and args.lfer:
        p.error("--ref and --lfer are mutually exclusive.  Use --lfer when a calibration is available.")

    results = run_calculation(
        msf_path     = Path(args.msf),
        ref_msf      = Path(args.ref)  if args.ref  else None,
        lfer_json    = Path(args.lfer) if args.lfer else None,
        orca_binary  = args.orca,
        nprocs       = args.nprocs,
        nconf        = args.nconf,
        ewindow      = args.ewindow,
        method       = args.method,
        do_tautomers = not args.no_tautomers,
        ph_min       = args.ph_min,
        ph_max       = args.ph_max,
        ph_step      = args.ph_step,
        temperature  = args.temp,
        dry_run      = args.dry_run,
        out_dir      = Path(args.outdir),
    )

    print(f"\nDone — {Path(args.msf).stem}")
    print("Predicted pKa values:")
    for k, pka in enumerate(results['pka_pred']):
        print(f"  pKa{k+1} = {pka:.3f}" if not np.isnan(pka) else f"  pKa{k+1} = nan")


if __name__ == '__main__':
    main()