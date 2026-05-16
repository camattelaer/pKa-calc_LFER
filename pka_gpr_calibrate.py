#!/usr/bin/env python3
"""
pka_gpr_calibrate.py  —  Descriptor-based GPR pKa calibration
==============================================================
Trains a Gaussian Process Regression (GPR) model to predict the DFT→pKa
correction using physical descriptors extracted directly from the ORCA
GFN2-xTB OPT+FREQ outputs that pka_calibrate.py already produces.

No functional group label is needed — the descriptors encode both the
chemical environment and the charge-state dependence implicitly.

Descriptors (extracted at zero extra computational cost)
---------------------------------------------------------
  x1  pKa_calc   = ΔG_DFT / (RT ln10)              [~190–215, encodes charge + chemistry]
  x2  q_acid     = formal charge of the acid species  [integer, Born solvation baseline]
  x3  Δq_O       = Σ q_Mulliken(all O atoms, base) − Σ q_Mulliken(all O atoms, acid)
  x4  Δq_N       = Σ q_Mulliken(all N atoms, base) − Σ q_Mulliken(all N atoms, acid)
  x5  Δq_S       = Σ q_Mulliken(all S atoms, base) − Σ q_Mulliken(all S atoms, acid)
                   [element-summed; invariant to conformer, rotation, and tautomers]
  x4  Δ|μ|       = |μ(base)| − |μ(acid)|              [Debye; polarity change]
  x5  ΔE_HOMO    = ε_HOMO(base) − ε_HOMO(acid)        [eV; electron availability change]

Physical interpretation
-----------------------
  x1 captures the raw DFT energy scale — the dominant ~200 unit offset.
  x2 captures the Born solvation q² term — the charge-state variation.
  x3 distinguishes functional groups: O-centred (~−0.25 e) vs N (~−0.40 e)
     vs S (~−0.20 e) changes.  This replaces the explicit fg label.
  x4 and x5 provide additional electronic context for the correction.

Why GPR
-------
  - Gives calibrated uncertainty σ(pKa) alongside every prediction.
  - Works well with 30–100 training points (our regime).
  - The RBF kernel encodes the assumption that chemically similar
    compounds need similar corrections — physically motivated.
  - Falls back to the prior mean (≈ mean correction) for unseen
    chemical space, rather than extrapolating wildly.

Directory and file conventions (compatible with pka_calibrate.py)
-----------------------------------------------------------------
  Output directory:        --outdir  (default: pka_results, same as calibrate)
  G_eff cache:             <outdir>/<compound>_geff.json          (shared)
  Descriptor cache:        <outdir>/<compound>_descriptors.json   (new)
  ORCA freq outputs:       <outdir>/work_<compound>/<ms_name>/s<NNN>/<label>_freq.out
  GPR model output:        <outdir>/gpr_model.json
  GPR report:              <outdir>/gpr_report.txt
  GPR diagnostic plot:     <outdir>/gpr_calibration_plot.png

Usage
-----
  # Train on same .msf files as LFER calibration (caches reused):
  python pka_gpr_calibrate.py training_set/*.msf \\
      --orca /apps/software/orca/orca-6.1.0/orca --nprocs 12

  # Point to existing work directory if calculations are done:
  python pka_gpr_calibrate.py training_set/*.msf --outdir pka_results

  # Check model coverage and stats:
  python pka_gpr_calibrate.py --check pka_results/gpr_model.json

  # Dry-run (mock descriptors, for testing):
  python pka_gpr_calibrate.py training_set/*.msf --dry-run

Integration with pka_calc.py
-----------------------------
  python pka_calc.py molecule.msf \\
      --gpr pka_results/gpr_model.json \\
      --orca /apps/software/orca/orca-6.1.0/orca --nprocs 12

  The molecule.msf does NOT need functional_group annotations.
  The GPR reads descriptors from the ORCA outputs it computes.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pka_calc import (
    Microstate, parse_msf, run_ensemble, _mock_run, _clean_xyz,
    KB_EV, EH_TO_EV, EH_TO_KCAL, LN10, TEMPERATURE,
    _parse_sp_energy, _parse_gibbs_and_sp,
    load_cached_G_eff, save_G_eff_cache, run_orca_for_compound,
)
try:
    from pka_calc import SCFDivergenceError
except ImportError:
    class SCFDivergenceError(Exception):  # type: ignore[no-redef]
        pass
try:
    from pka_calc import MissingThermoError
except ImportError:
    class MissingThermoError(Exception):  # type: ignore[no-redef]
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Physical descriptor data structures
# ──────────────────────────────────────────────────────────────────────────────

def _compute_ecfp4(msf_path: Path,
                   n_bits: int = 1024,
                   radius: int = 2) -> Optional[np.ndarray]:
    """
    Generate ECFP4 (Morgan r=2) fingerprint bit-vector for the compound in msf_path.

    Uses the microstate with lowest |charge| (best rdDetermineBonds convergence).
    Falls back through rdDetermineBonds parameter combinations.
    Returns float32 array of length n_bits, or None on failure.

    For compounds where fingerprint generation fails, None is returned and the
    caller substitutes a zero vector so the Tanimoto factor defaults to 1.0
    (pure Matérn — no structural information used).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError:
        return None

    try:
        mss = parse_msf(msf_path)
    except Exception:
        return None

    candidates = sorted(mss, key=lambda m: (abs(m.charge), -m.n_protons))
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)

    for ms in candidates:
        lines = [l for l in ms.xyz_block.strip().splitlines() if len(l.split()) == 4]
        if not lines:
            continue
        xyz = f"{len(lines)}\n\n" + "\n".join(lines)
        for charge in [ms.charge, 0]:
            for hueckel in [False, True]:
                try:
                    mol = Chem.MolFromXYZBlock(xyz)
                    if mol is None:
                        continue
                    rdDetermineBonds.DetermineBonds(
                        mol, charge=charge,
                        allowChargedFragments=True,
                        embedChiral=False,
                        useHueckel=hueckel,
                    )
                    mol_noh = Chem.RemoveHs(mol)
                    return gen.GetFingerprintAsNumPy(mol_noh).astype(np.float32)
                except Exception:
                    continue
    return None


def _tanimoto_matrix(fps: np.ndarray) -> np.ndarray:
    """
    N×N Tanimoto similarity matrix from (N, B) binary fingerprint matrix.
    T(a,b) = dot(a,b) / (|a|+|b|-dot(a,b)).
    Zero vectors → T=1.0 (no structural info → rely on Matérn only).
    """
    dot   = fps @ fps.T
    norms = fps.sum(axis=1, keepdims=True)
    union = norms + norms.T - dot
    return np.where(union > 0, dot / union, 1.0).astype(np.float64)


def _tanimoto_vector(fp_test: np.ndarray, fps_train: np.ndarray) -> np.ndarray:
    """(N_train,) Tanimoto similarities between test fp and each training fp."""
    dot   = fps_train @ fp_test
    union = fps_train.sum(axis=1) + fp_test.sum() - dot
    return np.where(union > 0, dot / union, 1.0).astype(np.float64)


@dataclass
class MicrostateDescriptors:
    """
    Physical descriptors for one microstate, extracted from GFN2-xTB OPT+FREQ.

    Charges are summed per element (not per atom), making descriptors
    invariant to:
      - GOAT conformer reordering / rotation / translation
      - Tautomers (e.g. imidazole N1-H vs N3-H give the same total Δq_N)
      - Resonance delocalisation (both oxygens of carboxylate contribute)

    All values are Boltzmann-weighted over the kept conformer ensemble,
    consistent with G_eff computation.
    """
    name:    str
    # Total Mulliken charge summed over all atoms of each element
    q_O:     Optional[float] = None   # sum q(all O atoms)
    q_N:     Optional[float] = None   # sum q(all N atoms)
    q_S:     Optional[float] = None   # sum q(all S atoms)
    q_C:     Optional[float] = None   # sum q(all C atoms)
    # Global electronic properties
    dipole:   Optional[float] = None  # total dipole magnitude, Debye
    homo_ev:  Optional[float] = None  # HOMO energy, eV
    lumo_ev:  Optional[float] = None  # LUMO energy, eV (first unoccupied orbital)
    # CPCM solvation energy
    e_cpcm_eh: Optional[float] = None  # CPCM dielectric correction, Eh
    # Mayer bond orders — populated when Print[P_BondOrders] 1 is in the DFT input.
    # Stored as a dict {(i,j): bo} for the best (lowest energy) conformer.
    # None for outputs generated before the print flag was added.
    mayer_bos: Optional[dict] = None
    # CHELPG per-atom charges — populated when %chelpg block is in the DFT input.
    # List indexed by atom number matching the xyz block ordering.
    # None for outputs generated before the CHELPG block was added.
    chelpg_charges: Optional[list] = None
    # Diagnostics
    n_conf:  int = 0



# ──────────────────────────────────────────────────────────────────────────────
# Functional-group indicator lookup table
# Maps functional_group annotation → (is_O_dominant, is_N_dominant, is_S_dominant).
# Used by StepDescriptors._fg_indicators() when annotation is available.
# Module-level (not inside dataclass) to avoid the mutable-default restriction.
# ──────────────────────────────────────────────────────────────────────────────
_FG_INDICATORS: dict[str, tuple[float, float, float]] = {
    # O-dominant
    "carboxylate":        (1.0, 0.0, 0.0),
    "phenol":             (1.0, 0.0, 0.0),
    "phosphate":          (1.0, 0.0, 0.0),
    "sulfonamide":        (0.5, 0.0, 0.5),  # S(=O)₂NH: shared O/S reorganisation
    # N-dominant
    "ammonium":           (0.0, 1.0, 0.0),
    "imidazolium":        (0.0, 1.0, 0.0),
    "pyridinium":         (0.0, 1.0, 0.0),
    "guanidinium":        (0.0, 1.0, 0.0),
    "dihydropyridinium":  (0.0, 1.0, 0.0),
    # S-dominant
    "thiol":              (0.0, 0.0, 1.0),
}


@dataclass
class StepDescriptors:
    """
    Descriptors for one ionisation step, ready for the GPR.

    Feature vector (6-dim, v6):
        [δpKa_calc, q_acid, Δq_O, Δq_N, Δq_C, Σ|Δq|]

    Feature selection history and rationale:
        v1–v5: progressively added/removed FG indicators, orbital features,
               dipole moment, and ΔE_LUMO/HOMO/gap/Δ|μ|.
        v6:    All orbital and dipole features removed from the GPR vector.
               Empirical finding at N=86 (bound=10.0):
                 ΔE_LUMO, ΔE_HOMO, ΔE_gap, Δ|μ| → all hit ls=10.0 (wall-bound)
               These have marginal correlation with the correction (r=+0.51 for
               ΔE_LUMO) but near-zero PARTIAL correlation once δpKa_calc is in
               the kernel. Both features encode deprotonation energy — ΔE_LUMO
               is redundant with δpKa_calc in the GPR kernel sense.
               Increasing ls_max beyond 10.0 doesn't help: at ls=10 the kernel
               similarity is already 0.9999 per 1σ — functionally indistinguishable
               from ls=∞. The amplitude absorbs the constant contribution.
               The 4 orbital/dipole fields remain stored in StepDescriptors and
               serialised to JSON for use by SVR or other non-ARD models.

    N/D = 86/6 = 14.3 — well above the reliable threshold.
    """
    compound:    str
    acid_name:   str
    base_name:   str
    acid_charge: int
    pka_calc:    float
    pka_exp:     float
    # Element-summed Mulliken charge differences (base − acid)
    dq_O:        Optional[float] = None
    dq_N:        Optional[float] = None
    dq_S:        Optional[float] = None
    dq_C:        Optional[float] = None
    # Global electronic descriptors
    d_dipole:    Optional[float] = None
    d_homo:      Optional[float] = None
    d_lumo:      Optional[float] = None
    d_gap:       Optional[float] = None
    # CPCM solvation energy difference
    d_cpcm:      Optional[float] = None
    # Mayer bond order descriptors (None for old outputs without Print[P_BondOrders])
    dbo_xc:      Optional[float] = None   # ΔBO of ionisable-heteroatom–C bond
    dbo_max:     Optional[float] = None   # max |ΔBO| anywhere in molecule
    dbo_ring:    Optional[float] = None   # max |ΔBO| in aromatic ring bonds
    # CHELPG electrostatic potential descriptors (None for old outputs without %chelpg)
    q_chelpg_site_acid: Optional[float] = None  # CHELPG charge at ionisable atom in acid
    dq_chelpg_site:     Optional[float] = None  # CHELPG charge change at ionisable atom (base - acid)
    # ECFP4 fingerprint for Tanimoto product kernel (None if generation failed)
    fp_bits:     Optional[np.ndarray] = None
    # Structural / topological descriptors
    n_heavy:     int = 0
    n_O:         int = 0
    n_N:         int = 0
    n_S:         int = 0

    FEATURE_NAMES  = ["δpKa_calc", "q_acid", "ΔE_CPCM",
                       "ΔBO_XC", "ΔBO_ring", "q_CHELPG_site"]
    FALLBACK_NAMES = ["δpKa_calc", "q_acid"]

    # functional_group annotation from MSF base microstate (set by build_step_descriptors)
    functional_group: Optional[str] = None

    def _sum_abs_dq(self) -> float:
        """Total charge reorganisation: Σ|Δq| over all tracked elements."""
        return (abs(self.dq_O or 0.0) + abs(self.dq_N or 0.0)
                + abs(self.dq_S or 0.0) + abs(self.dq_C or 0.0))

    def feature_vector(self, charge_means: Optional[dict] = None) -> Optional[np.ndarray]:
        """
        Return 7-dim feature vector or None on parse failure.

        [δpKa_calc, q_acid, Δq_C, ΔE_CPCM, ΔBO_XC, ΔBO_ring, ΔE_LUMO]

        Feature selection — v10 (current)
        ----------------------------------
        Previous v9 dropped all frontier orbital features. This version
        reinstates ΔE_LUMO as a candidate for two reasons:

          1. It had the second-highest marginal correlation with the DFT
             correction at N=142 (r=+0.44, just below q_acid r=+0.45).

          2. Its conditional independence from the other features may have
             changed now that ΔBO_ring is in the model. The previous
             wall-binding was established with Δq_O/Δq_N/f_O etc. in the
             model — a different feature space. We let ARD decide again.

          3. ΔE_LUMO and ΔBO_ring are physically distinct:
               ΔE_LUMO   — property of the ACID before deprotonation;
                            encodes how EW/ED substituents lower/raise the
                            LUMO of the acidic form. Proxy for Hammett σ.
               ΔBO_ring  — RESPONSE after deprotonation; encodes how much
                            charge the ring accepts. Proxy for Hammett ρ·σ.
             Both can be independently informative for substituent effects.

          If ΔE_LUMO walls again (ls→10.0) at this N, it confirms genuine
          collinearity with the new 6-feature basis and should be dropped.

        N/D = 142/7 = 20.3.
        """
        if self.dq_C is None and self.dbo_xc is None and self.dbo_ring is None:
            if self.dq_O is None and self.dq_N is None and self.dq_S is None:
                return None
        pka_feat = (self.pka_calc - charge_means[self.acid_charge]
                    if charge_means and self.acid_charge in charge_means
                    else self.pka_calc)
        return np.array([
            pka_feat,
            float(self.acid_charge),
            self.d_cpcm   if self.d_cpcm   is not None else 0.0,
            self.dbo_xc   if self.dbo_xc   is not None else 0.0,
            self.dbo_ring if self.dbo_ring is not None else 0.0,
            self.q_chelpg_site_acid if self.q_chelpg_site_acid is not None else 0.0,
        ])

    def feature_vector_partial(self, charge_means: Optional[dict] = None) -> np.ndarray:
        """[δpKa_calc, q_acid] — always available, used as fallback."""
        pka_feat = (self.pka_calc - charge_means[self.acid_charge]
                    if charge_means and self.acid_charge in charge_means
                    else self.pka_calc)
        return np.array([pka_feat, float(self.acid_charge)])

# ──────────────────────────────────────────────────────────────────────────────
# ORCA output parsers for descriptors
# ──────────────────────────────────────────────────────────────────────────────

def _parse_mulliken_charges(out_txt: str) -> Optional[list[float]]:
    """
    Parse Mulliken atomic charges from a GFN2-xTB ORCA output.
    Returns list indexed by atom index (0-based), or None if not found.
    """
    # ORCA writes:
    # MULLIKEN ATOMIC CHARGES
    # -----------------------
    #    0 C  :   -0.1234
    #    1 N  :    0.4567
    m_block = re.search(
        r'MULLIKEN ATOMIC CHARGES\s*[-]+\s*((?:\s*\d+\s+\w+\s*:\s*[-\d.]+\s*\n)+)',
        out_txt, re.MULTILINE,
    )
    if not m_block:
        return None
    charges = []
    for line in m_block.group(1).splitlines():
        m = re.match(r'\s*\d+\s+\w+\s*:\s*([-\d.]+)', line)
        if m:
            charges.append(float(m.group(1)))
    return charges if charges else None


def _parse_dipole_debye(out_txt: str) -> Optional[float]:
    """
    Parse total dipole moment magnitude in Debye.
    Tries several ORCA output formats.
    """
    # Format 1: "Magnitude (Debye)  : 1.2345"
    m = re.search(r'Magnitude\s*\(Debye\)\s*[:\s]+([\d.]+)', out_txt)
    if m:
        return float(m.group(1))
    # Format 2: "Total Dipole Moment    :     X      Y      Z   |Dip|"
    #           "                         0.00   0.00   1.23   1.23"
    # — the last number on the coordinate line
    m = re.search(
        r'Total Dipole Moment.*\n\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([\d.]+)',
        out_txt,
    )
    if m:
        return float(m.group(1))
    # Format 3: GFN2-xTB specific "Dipole moment (a.u.):" followed by XYZ and |μ|
    m = re.search(r'Dipole moment.*?:\s*\n.*?\|\s*([\d.]+)\s*\|', out_txt, re.DOTALL)
    if m:
        return float(m.group(1)) * 2.5418  # convert a.u. → Debye
    # Format 4: " dipole moment (a.u.)\n   x= ... y= ... z= ... total= X.XX"
    m = re.search(r'total\s*=\s*([\d.]+)\s*Debye', out_txt, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _parse_homo_ev(out_txt: str) -> Optional[float]:
    """
    Parse HOMO energy in eV from the ORBITAL ENERGIES block.
    Returns the energy of the highest doubly-occupied orbital.
    Works on both DFT (NO   OCC   E(Eh)   E(eV)) and GFN2-xTB
    (index   occ   E(Eh)   E(eV)  (HOMO)) formats.
    """
    homo_ev = None
    # DFT format: last line with OCC = 2.0000
    for m in re.finditer(
        r'^\s*\d+\s+(2\.0+)\s+[-\d.]+\s+([-\d.]+)',
        out_txt, re.MULTILINE,
    ):
        homo_ev = float(m.group(2))
    return homo_ev


def _parse_lumo_ev(out_txt: str) -> Optional[float]:
    """
    Parse LUMO energy in eV from the ORBITAL ENERGIES block.
    Returns the energy of the lowest unoccupied orbital (OCC = 0.0000).
    Only meaningful in the DFT _dft.out; not available from GFN2-xTB _freq.out.
    """
    # Find the ORBITAL ENERGIES section, then locate the first 0.0000-occupied line
    # that follows a 2.0000-occupied line (i.e. the LUMO, not a virtual below HOMO)
    in_block = False
    last_was_occ = False
    for line in out_txt.splitlines():
        if 'ORBITAL ENERGIES' in line:
            in_block = True
            continue
        if not in_block:
            continue
        m = re.match(r'^\s*\d+\s+([\d.]+)\s+[-\d.]+\s+([-\d.]+)', line)
        if m:
            occ = float(m.group(1))
            ev  = float(m.group(2))
            if occ >= 1.9:
                last_was_occ = True
            elif occ < 0.1 and last_was_occ:
                return ev   # first virtual after last occupied = LUMO
        elif in_block and line.strip() == '' and last_was_occ:
            break   # end of orbital block
    return None


def _parse_e_cpcm(out_txt: str) -> Optional[float]:
    """
    Parse the CPCM dielectric contribution from ORCA DFT output (Eh).

    ORCA 6.x prints the CPCM correction in the TOTAL SCF ENERGY block:

        Total Energy       :       -398.37875299 Eh
        ...
        CPCM Dielectric    :         -0.14191866 Eh

    The "CPCM Dielectric" line is the dielectric polarisation correction —
    the energy gained by embedding the solute charge density in the implicit
    solvent.  It is always negative (stabilising) and its magnitude grows
    with the square of the charge on the molecule.

    Physical use as a descriptor:
        d_cpcm = CPCM_Dielectric(base) − CPCM_Dielectric(acid)
        This encodes how much MORE the solvent stabilises the deprotonated
        form vs the protonated form.  For a carboxylate (charge −1 → −2),
        the base gains more solvation than an ammonium (charge +1 → 0).
        This single number encodes both the charge magnitude AND the spatial
        charge distribution, making it more informative than either q_acid
        or the Δq features alone.

    Note: the descriptor is the CPCM *correction* only (not the total energy),
    so gas-phase DFT errors cancel in the difference d_cpcm = base − acid.

    Patterns tried in order of ORCA version:
      1. ORCA 6.x:  'CPCM Dielectric    :    -X.XXXX Eh'  ← actual format
      2. ORCA 5.0+: 'Total Energy corrected for CPCM : -X.XXXX Eh'
      3. ORCA 4/5:  'E(PCM)         =   -X.XXXX'
      4. ORCA 4.x:  'Total energy (with pcm corr)  = -X.XXXX'

    Returns the value in Hartree (Eh), or None if not found.
    """
    patterns = [
        # ORCA 6.x — dielectric correction in TOTAL SCF ENERGY block
        r'CPCM Dielectric\s*:\s*([-\d.]+)',
        # ORCA 5.x — total energy with CPCM correction
        r'Total Energy corrected for CPCM\s*:\s*([-\d.]+)',
        # ORCA 4/5 — E(PCM) single-point block
        r'E\(PCM\)\s*=\s*([-\d.]+)',
        # Older ORCA 4.x
        r'Total energy \(with pcm corr\)\s*=\s*([-\d.]+)',
    ]
    for pat in patterns:
        m = re.search(pat, out_txt)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _parse_chelpg_charges(out_txt: str) -> Optional[list[float]]:
    """
    Parse per-atom CHELPG charges from an ORCA DFT output.

    ORCA prints (when %chelpg block is present):

        CHELPG Charges
        --------------
        0 C :    -0.231456
        1 O :     0.541234
        2 H :     0.123456
        ...

        Total charge:    0.000000

    Returns a list of floats indexed by atom number (same ordering as the
    xyz block in the input), or None if the block is absent (old outputs
    without %chelpg in the input).
    """
    m_block = re.search(
        r'CHELPG Charges\s*\n[-\s]+\n(.*?)(?=\n\s*Total charge|\n\n|\Z)',
        out_txt, re.DOTALL | re.IGNORECASE
    )
    if not m_block:
        return None
    charges = []
    for line in m_block.group(1).splitlines():
        m = re.match(r'\s*(\d+)\s+\w+\s*:\s*([-\d.]+)', line)
        if m:
            idx = int(m.group(1))
            q   = float(m.group(2))
            # Fill any gaps (shouldn't happen, but defensive)
            while len(charges) <= idx:
                charges.append(0.0)
            charges[idx] = q
    return charges if charges else None


def _chelpg_descriptors(
        acid_xyz:       str,
        base_xyz:       str,
        acid_chelpg:    Optional[list[float]],
        base_chelpg:    Optional[list[float]],
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute two CHELPG-based descriptors from acid/base pairs.

    Returns (q_chelpg_site_acid, dq_chelpg_mean) or (None, None).

    q_chelpg_site_acid
        CHELPG charge on the ionisable heteroatom in the acid microstate.
        Encodes the full ESP at the deprotonation site — integrating inductive
        effects from every substituent (ionisable or not) through bonds AND
        through space. Remote groups (COO- in cetirizine, CF3 in fluoxetine,
        C=O amide in lidocaine) all modulate this charge even without direct
        bonding to the ionisable N/O/S.

        More informative than element-summed Δq_N because:
          - It is a per-atom absolute value, not summed across all N atoms
          - It reflects the PRE-deprotonation state (the reactant), which
            determines the proton affinity, not the product stabilisation
          - It is fit to the molecular ESP surface (SASA proxy), capturing
            through-space effects that are absent from Hirshfeld partitioning

    dq_chelpg_mean
        Mean CHELPG charge change (base − acid) averaged over all heavy atoms
        EXCLUDING the ionisable H (which disappears) and the ionisable atom
        itself (captured by q_chelpg_site_acid). This encodes how the
        deprotonation redistributes charge over the REST of the molecule —
        the difference between resonance delocalisation (large spread,
        e.g. picrate) and localised charge retention (small spread, e.g.
        aliphatic carboxylate or simple ammonium).
    """
    if acid_chelpg is None or base_chelpg is None:
        return None, None

    def parse_atoms(xyz_block):
        atoms = []
        for line in xyz_block.strip().splitlines():
            parts = line.split()
            if len(parts) == 4:
                try:
                    atoms.append((parts[0],
                                  float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    pass
        return atoms

    acid_atoms = parse_atoms(acid_xyz)
    base_atoms = parse_atoms(base_xyz)
    if not acid_atoms or not base_atoms:
        return None, None

    # ── q_chelpg_site_acid ────────────────────────────────────────────────────
    ion_idx = _identify_ionisable_atom(acid_xyz, base_xyz)
    q_site = None
    if ion_idx is not None and ion_idx < len(acid_chelpg):
        q_site = acid_chelpg[ion_idx]

    # ── dq_chelpg_mean ────────────────────────────────────────────────────────
    # Identify the removed H in the acid: the H bonded to the ionisable atom
    # that is absent in the base. We find it by checking bonded H atoms of
    # the ionisable atom that don't have a counterpart in the base.
    #
    # Since the heavy atoms have the same ordering in acid and base (GOAT
    # preserves heavy atom ordering), and the removed H is the LAST H in
    # the acid bonded to the ionisable site (ORCA appends H atoms at the end
    # by convention), we can match heavy atoms by index directly.
    COV_RADII = {'H':0.31,'C':0.76,'N':0.71,'O':0.66,'S':1.05,'P':1.07,
                 'F':0.57,'Cl':1.02,'Br':1.20,'I':1.39,'B':0.84}

    # Build correspondence: heavy atom index in acid → index in base
    # (same position, since only one H is removed and heavy atoms are invariant)
    acid_heavy_idx = [i for i,(s,*_) in enumerate(acid_atoms) if s != 'H']
    base_heavy_idx = [i for i,(s,*_) in enumerate(base_atoms) if s != 'H']

    # Δq_CHELPG_site: charge change at the ionisable atom (base - acid).
    # Physically distinct from q_CHELPG_site (absolute pre-deprotonation charge)
    # and from ΔBO_XC (bond order change at X-C):
    #   - phenolate:        O becomes strongly negative (large negative Δq_site)
    #   - carboxylate:      charge delocalises across 2 O → smaller per-atom Δq
    #   - ammonium:         N gains significant electron density (large Δq_site)
    #   - imidazolium:      charge shared across N1 and N3 → intermediate Δq_site
    # This encodes the localisation/delocalisation of the new lone pair,
    # which is orthogonal to the bond order change (geometry of electron density)
    # and to the pre-deprotonation environment (q_CHELPG_site).
    dq_site = None
    if (ion_idx is not None
            and ion_idx < len(acid_chelpg if acid_chelpg else [])
            and len(base_heavy_idx) == len(acid_heavy_idx)):
        # Find the corresponding index in the base (same heavy-atom rank)
        acid_rank = [i for i,(s,*_) in enumerate(acid_atoms) if s != 'H'].index(ion_idx)                     if ion_idx in [i for i,(s,*_) in enumerate(acid_atoms) if s != 'H'] else None
        if (acid_rank is not None
                and acid_rank < len(base_heavy_idx)
                and base_heavy_idx[acid_rank] < len(base_chelpg if base_chelpg else [])):
            bi = base_heavy_idx[acid_rank]
            dq_site = base_chelpg[bi] - acid_chelpg[ion_idx]

    return q_site, dq_site


def _rerun_dft_if_needed(
        ms:          "Microstate",
        ms_dir:      Path,
        orca_binary: str,
        nprocs:      int,
        method:      str,
        solvent_dft: str,
) -> None:
    """
    For each conformer in ms_dir, check whether the DFT single-point was run
    with the current input template by inspecting the embedded input in the
    *_dft.out file.  If the %chelpg block is absent, rerun the DFT SP only —
    GOAT and xTB FREQ are not touched.

    Geometry source (in priority order):
      1. Optimised xyz extracted from *_freq.out (the xTB-optimised geometry)
      2. xyz embedded in the old *_dft.out itself (fallback)

    The updated *_dft.inp and *_dft.out overwrite the old files in-place so
    subsequent runs see the new input and skip the rerun automatically.
    """
    from pka_calc import (
        _dft_inp_is_current, _dft_sp_inp, _run_orca,
        _read_optimised_xyz, _clean_xyz, _parse_sp_energy
    )

    if orca_binary is None:
        return   # dry-run mode

    for sdir in sorted(ms_dir.glob("s*")):
        dft_outs = sorted(sdir.glob("*_dft.out"))
        if not dft_outs:
            continue
        d_out = dft_outs[0]
        d_inp = d_out.with_suffix('.inp')
        label = d_out.stem.replace("_dft", "")

        # Check the ACTUAL input that was run (embedded in .out)
        if _dft_inp_is_current(d_inp, ms, "", nprocs, method, solvent_dft):
            continue   # already has CHELPG — skip

        logging.info("  [%s/%s] DFT output predates %%chelpg block. "
                     "Rerunning DFT SP…", ms.name, label)

        # --- Get geometry ---
        # Prefer xTB-optimised geometry from freq.out
        opt_xyz = None
        freq_outs = sorted(sdir.glob("*_freq.out"))
        if freq_outs:
            opt_xyz = _read_optimised_xyz(freq_outs[0])

        # Fallback: extract xyz from the embedded input in the old dft.out
        if not opt_xyz:
            old_txt = d_out.read_text(errors='replace')
            m_inp = re.search(
                r'NAME\s*=\s*\S+\s*\n(.*?)\*\*\*\*END OF INPUT\*\*\*\*',
                old_txt, re.DOTALL
            )
            if m_inp:
                raw_lines = [re.sub(r'^\|\s*\d+>\s?', '', l)
                             for l in m_inp.group(1).splitlines()]
                raw_inp = '\n'.join(raw_lines)
                # Extract xyz block between "* xyz charge mult" and "*"
                m_xyz = re.search(r'\*\s*xyz\s+[-\d]+\s+\d+\s*\n(.*?)\n\*',
                                  raw_inp, re.DOTALL)
                if m_xyz:
                    opt_xyz = m_xyz.group(1).strip()

        if not opt_xyz:
            logging.warning("  [%s/%s] Cannot recover geometry for DFT rerun. "
                            "Skipping.", ms.name, label)
            continue

        # --- Rerun ---
        new_inp = _dft_sp_inp(ms, opt_xyz, sdir, label, nprocs, method, solvent_dft)
        try:
            _run_orca(new_inp, orca_binary)
            _parse_sp_energy(d_out)   # validate clean completion
            logging.info("  [%s/%s] DFT rerun with %%chelpg complete.", ms.name, label)
        except Exception as e:
            logging.warning("  [%s/%s] DFT rerun failed: %s", ms.name, label, e)


def _parse_mayer_bond_orders(out_txt: str) -> Optional[dict[tuple[int,int], float]]:
    """
    Parse Mayer bond orders from an ORCA DFT output.

    ORCA prints Mayer bond orders by default (P_Mayer = 1 is the default),
    so ALL existing *_dft.out files already contain this block — no new
    ORCA calculations are needed.

    Actual ORCA 6.x format (multiple bonds packed per line, colon separator):

        Mayer bond orders larger than 0.100000
        B(  0-N ,  1-C ) :   0.8266 B(  0-N ,  7-H ) :   0.7380 B(  0-N ,  8-H ) :   0.8647
        B(  0-N ,  9-H ) :   0.8586 B(  1-C ,  2-C ) :   0.9199 ...
        B(  2-C ,  3-O ) :   1.7305 B(  2-C ,  4-O ) :   1.6368
        ...
        [blank line terminates block]

    Returns dict {(i, j): bo} with i < j, or None if block absent.
    """
    # Find the header line — threshold value varies (0.1 or 0.100000)
    m_start = re.search(r'Mayer bond orders larger than', out_txt)
    if not m_start:
        return None

    # Extract everything from the line after the header until a blank line
    block_start = out_txt.find('\n', m_start.start()) + 1
    block_end   = out_txt.find('\n\n', block_start)
    if block_end == -1:
        block_end = len(out_txt)
    block = out_txt[block_start:block_end]

    # Each bond entry: B(  i-SYM ,  j-SYM ) :   value
    # Multiple entries can appear on the same line
    bos: dict[tuple[int,int], float] = {}
    pattern = re.compile(r'B\(\s*(\d+)-\w+\s*,\s*(\d+)-\w+\s*\)\s*:\s*([-\d.]+)')
    for m in pattern.finditer(block):
        i, j, bo = int(m.group(1)), int(m.group(2)), float(m.group(3))
        bos[(min(i,j), max(i,j))] = bo

    return bos if bos else None


def _identify_ionisable_atom(acid_xyz: str, base_xyz: str) -> Optional[int]:
    """
    Find the 0-based index (in the full atom list including H) of the heavy
    atom that loses a proton going from acid to base.

    Strategy: compare H-count bonded to each heavy atom in acid vs base.
    The heavy atom with exactly one fewer bonded H in the base is the site.
    Works without any atom-type knowledge — just distance-based bonding.
    Heavy-atom ordering is assumed identical in acid and base (guaranteed by
    GOAT/GFN2-xTB which preserves atom ordering across conformers).
    """
    COV_RADII = {
        'H':0.31,'C':0.76,'N':0.71,'O':0.66,'S':1.05,'P':1.07,
        'F':0.57,'Cl':1.02,'Br':1.20,'I':1.39,'B':0.84,
    }

    def parse_atoms(xyz_block):
        atoms = []
        for line in xyz_block.strip().splitlines():
            parts = line.split()
            if len(parts) == 4:
                try:
                    atoms.append((parts[0],
                                  float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    pass
        return atoms

    def count_bonded_H(atoms, idx):
        sym_i, xi, yi, zi = atoms[idx]
        r_i = COV_RADII.get(sym_i, 0.77)
        count = 0
        for j, (sym_j, xj, yj, zj) in enumerate(atoms):
            if j == idx or sym_j != 'H':
                continue
            d = ((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2) ** 0.5
            if d < r_i + 0.31 + 0.4:
                count += 1
        return count

    acid_atoms = parse_atoms(acid_xyz)
    base_atoms = parse_atoms(base_xyz)
    if not acid_atoms or not base_atoms:
        return None

    acid_heavy = [(i, s) for i,(s,*_) in enumerate(acid_atoms) if s != 'H']
    base_heavy = [(i, s) for i,(s,*_) in enumerate(base_atoms) if s != 'H']
    if len(acid_heavy) != len(base_heavy):
        return None

    for (ai, sym_a), (bi, sym_b) in zip(acid_heavy, base_heavy):
        if sym_a != sym_b:
            continue
        if count_bonded_H(acid_atoms, ai) - count_bonded_H(base_atoms, bi) == 1:
            return ai   # 0-based index in full (including H) atom list
    return None


def _bond_order_descriptors(
        acid_xyz:  str,
        base_xyz:  str,
        acid_bos:  Optional[dict],
        base_bos:  Optional[dict],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute three Mayer bond-order descriptors from acid/base pairs.

    Returns (dbo_xc, dbo_max, dbo_ring) — all None when bond orders are
    absent (old outputs without Print[P_BondOrders] 1).

    dbo_xc  — ΔBO of the bond between the ionisable heteroatom X and its
               directly bonded C. Encodes local resonance stabilisation:
                 phenolate        ΔBO(C–O) ≈ +0.20  (partial C=O develops)
                 4-nitrophenolate ΔBO(C–O) ≈ +0.30  (relay into ring)
                 aliphatic COO-   ΔBO(C–O) ≈ −0.05  (delocalises within COO-)
                 aliphatic NH2    ΔBO(C–N) ≈ −0.05  (no conjugation)

    dbo_max — largest |ΔBO| anywhere in the molecule. Captures long-range
               relay (charge propagation through NO2 or extended conjugation).

    dbo_ring — largest |ΔBO| among bonds between aromatic-type C/N atoms
               (BO ≈ 1.2–1.8 in acid). Zero for aliphatic compounds; large
               when deprotonation delocalises charge into a ring.
    """
    if acid_bos is None or base_bos is None:
        return None, None, None

    def parse_atoms(xyz_block):
        atoms = []
        for line in xyz_block.strip().splitlines():
            parts = line.split()
            if len(parts) == 4:
                try:
                    atoms.append((parts[0],
                                  float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    pass
        return atoms

    acid_atoms = parse_atoms(acid_xyz)
    if not acid_atoms:
        return None, None, None

    all_bonds = set(acid_bos) | set(base_bos)

    # ── dbo_xc ────────────────────────────────────────────────────────────────
    dbo_xc = None
    ion_idx = _identify_ionisable_atom(acid_xyz, base_xyz)
    if ion_idx is not None and ion_idx < len(acid_atoms):
        sym_ion = acid_atoms[ion_idx][0]
        if sym_ion in ('O', 'N', 'S', 'P', 'F'):
            xc_bonds = [
                (i, j) for (i, j) in all_bonds
                if (i == ion_idx and j < len(acid_atoms) and acid_atoms[j][0] == 'C')
                or (j == ion_idx and i < len(acid_atoms) and acid_atoms[i][0] == 'C')
            ]
            if xc_bonds:
                deltas = [
                    base_bos.get(b, 0.0) - acid_bos.get(b, 0.0)
                    for b in xc_bonds
                ]
                dbo_xc = max(deltas, key=abs)

    # ── dbo_max ───────────────────────────────────────────────────────────────
    dbo_max = max(
        abs(base_bos.get(b, 0.0) - acid_bos.get(b, 0.0))
        for b in all_bonds
    ) if all_bonds else None

    # ── dbo_ring ──────────────────────────────────────────────────────────────
    # Ring bonds: aromatic C/N–C/N with BO in [1.15, 1.85] in the acid
    ring_bonds = [
        (i, j) for (i, j), bo in acid_bos.items()
        if i < len(acid_atoms) and j < len(acid_atoms)
        and acid_atoms[i][0] in ('C','N') and acid_atoms[j][0] in ('C','N')
        and 1.15 <= bo <= 1.85
    ]
    dbo_ring = max(
        abs(base_bos.get(b, 0.0) - acid_bos.get(b, 0.0))
        for b in ring_bonds
    ) if ring_bonds else 0.0


    return dbo_xc, dbo_max, dbo_ring


def _parse_element_charges(
        out_txt: str,
        element_symbols: list[str],
) -> Optional[dict[str, float]]:
    """
    Parse atomic charges and sum them per element.

    Charge scheme preference order (best to worst for pKa descriptors):
      1. Hirshfeld — partition-based, basis-set independent, most consistent
         across different charge states and functional groups. Available from
         ORCA when Print[P_Hirshfeld] 1 is set (added to _dft_sp_inp).
      2. Mulliken — orbital-projection, basis-set dependent but universally
         available. Kept as primary fallback.
      3. Löwdin — symmetric orthogonalisation, somewhat more stable than
         Mulliken with diffuse basis sets.

    For the Δq descriptors used in GPR calibration, consistent relative
    differences between acid and base microstates matter more than absolute
    accuracy. Hirshfeld charges give more consistent Δq values across
    functional groups, improving GPR kernel distances between dissimilar
    chemical families.

    Handles both ORCA colon format ("0 N :  -0.412") and the older format
    without colon ("0 N  -0.412").

    Returns {element: total_charge} or None if no charge block found.
    """
    def _extract(header_re: str) -> Optional[dict[str, float]]:
        m_header = re.search(header_re, out_txt, re.IGNORECASE)
        if not m_header:
            return None
        per_element: dict[str, float] = {el: 0.0 for el in element_symbols}
        found_any = False
        for line in out_txt[m_header.end():].splitlines():
            m = re.match(r'^\s*(\d+)\s+([A-Za-z]+)\s*:?\s*([-\d.]+)', line)
            if not m:
                stripped = line.strip()
                if stripped == '' and found_any:
                    break
                if stripped.startswith('-' * 5):
                    continue
                if found_any and re.match(r'^[A-Z]{2}', stripped):
                    break
                continue
            sym    = m.group(2)
            charge = float(m.group(3))
            found_any = True
            if sym in per_element:
                per_element[sym] += charge
        return per_element if found_any else None

    # Try each charge scheme in preference order
    for header_re, scheme in [
        (r'HIRSHFELD CHARGES\s*\n\s*-+\s*\n',         "Hirshfeld"),
        (r'MULLIKEN ATOMIC CHARGES\s*\n\s*-+\s*\n',   "Mulliken"),
        (r'MULLIKEN CHARGES\s*\n\s*-+\s*\n',           "Mulliken"),
        (r'LOEWDIN ATOMIC CHARGES\s*\n\s*-+\s*\n',    "Loewdin"),
    ]:
        result = _extract(header_re)
        if result is not None:
            if scheme == "Loewdin":
                logging.debug("  Using Löwdin charges (Hirshfeld/Mulliken absent).")
            elif scheme == "Hirshfeld":
                logging.debug("  Using Hirshfeld charges.")
            return result
    return None


def _parse_descriptors_from_dft_out(
        dft_out: Path,
        element_symbols: list[str],
) -> tuple[Optional[dict[str, float]], Optional[float], Optional[float],
           Optional[float], Optional[float]]:
    """
    Parse (element_charges, dipole_Debye, homo_eV, lumo_eV, e_cpcm_Eh)
    from the DFT SP output (*_dft.out, r2SCAN-3c/CPCM).

    Returns (None, None, None, None, None) if file missing or unparseable.
    """
    if not dft_out.exists():
        return None, None, None, None, None
    txt    = dft_out.read_text(errors='replace')
    eq     = _parse_element_charges(txt, element_symbols)
    dipole = _parse_dipole_debye(txt)
    homo   = _parse_homo_ev(txt)
    lumo   = _parse_lumo_ev(txt)
    cpcm   = _parse_e_cpcm(txt)
    return eq, dipole, homo, lumo, cpcm

# ──────────────────────────────────────────────────────────────────────────────
# Descriptor cache
# ──────────────────────────────────────────────────────────────────────────────

def _desc_cache_path(msf_path: Path, out_dir: Path) -> Path:
    return out_dir / f"{msf_path.stem}_descriptors.json"


def load_cached_descriptors(
        msf_path: Path, out_dir: Path,
) -> Optional[dict[str, dict]]:
    """
    Load descriptor cache {ms_name: {q_O, q_N, q_S, q_C, dipole, homo_ev, ...}}.

    Rejects the cache and returns None if:
      - The file does not exist.
      - The cache was written in the old format (had 'q_ionising' key).
      - Every microstate in the cache has None for all element charges —
        indicating a prior parse failure that should be retried.
    """
    cache = _desc_cache_path(msf_path, out_dir)
    if not cache.exists():
        return None
    try:
        data = json.loads(cache.read_text())

        # Reject old format (pre-element-summed redesign)
        for v in data.values():
            if "q_ionising" in v:
                logging.info("  [%s] Old descriptor cache (q_ionising format) — will recompute.",
                             msf_path.stem)
                return None

        # Reject if all microstates have all-None element charges (prior parse failure)
        def _has_any_charge(v: dict) -> bool:
            return any(v.get(k) is not None for k in ("q_O", "q_N", "q_S"))

        if data and not any(_has_any_charge(v) for v in data.values()):
            logging.info("  [%s] Descriptor cache has no usable element charges — will recompute.",
                         msf_path.stem)
            return None

        logging.info("  [%s] Loaded cached descriptors from %s",
                     msf_path.stem, cache.name)
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def save_descriptor_cache(
        msf_path:    Path,
        descriptors: dict[str, dict],
        out_dir:     Path,
) -> None:
    cache = _desc_cache_path(msf_path, out_dir)
    cache.write_text(json.dumps(descriptors, indent=2))
    logging.info("  [%s] Descriptors cached → %s", msf_path.stem, cache.name)

# ──────────────────────────────────────────────────────────────────────────────
# Per-compound descriptor extraction
# ──────────────────────────────────────────────────────────────────────────────

def _elements_in_xyz(xyz_block: str) -> set[str]:
    """Return set of heavy-atom element symbols present in an xyz block."""
    elems = set()
    for line in xyz_block.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4:
            sym = parts[0]
            if sym != 'H':
                elems.add(sym)
    return elems


def _count_elements(xyz_block: str) -> dict[str, int]:
    """
    Return {element: count} for all atoms (including H) in an xyz block.
    Used to compute n_heavy, n_O, n_N, n_S for StepDescriptors.
    """
    counts: dict[str, int] = {}
    for line in xyz_block.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                float(parts[1])   # check second token is a coordinate
                sym = parts[0]
                counts[sym] = counts.get(sym, 0) + 1
            except ValueError:
                pass
    return counts


def extract_descriptors_for_compound(
        msf_path:    Path,
        out_dir:     Path,
        dry_run:     bool = False,
) -> dict[str, MicrostateDescriptors]:
    """
    Extract element-summed Mulliken charges, dipole, and HOMO for each
    microstate of a compound.

    Reads from ORCA freq output files in the standard layout:
        <out_dir>/work_<compound>/<ms_name>/s<NNN>/<label>_freq.out

    Design choices
    --------------
    Charges are summed per element (Δq_O, Δq_N, Δq_S), not per atom.
    This makes descriptors invariant to:
      • GOAT resampling across charge states (different conformer, same
        element composition → same total element charges)
      • Tautomers (N1-H vs N3-H imidazole: both give the same Σq_N)
      • Resonance delocalisation (both COO⁻ oxygens contribute to Σq_O)

    No atom-position matching or spatial geometry comparison is performed.

    Returns {ms_name: MicrostateDescriptors}.
    """
    compound = msf_path.stem
    work_dir = out_dir / f"work_{compound}"

    # ── Load cache ────────────────────────────────────────────────────────────
    cached = load_cached_descriptors(msf_path, out_dir)
    if cached is not None:
        result = {}
        for name, d in cached.items():
            md         = MicrostateDescriptors(name=name)
            md.q_O     = d.get("q_O")
            md.q_N     = d.get("q_N")
            md.q_S     = d.get("q_S")
            md.q_C     = d.get("q_C")
            md.dipole     = d.get("dipole")
            md.homo_ev    = d.get("homo_ev")
            md.lumo_ev    = d.get("lumo_ev")
            md.e_cpcm_eh  = d.get("e_cpcm_eh")
            md.n_conf     = d.get("n_conf", 0)
            # Mayer bond orders: stored as [[i, j, bo], ...] list
            raw_bos = d.get("mayer_bos")
            if raw_bos:
                md.mayer_bos = {(int(r[0]), int(r[1])): float(r[2]) for r in raw_bos}
            # CHELPG charges: stored as flat list
            raw_chelpg = d.get("chelpg_charges")
            if raw_chelpg:
                md.chelpg_charges = [float(q) for q in raw_chelpg]
            result[name] = md
        return result

    # ── Dry-run: mock descriptors chemically plausible by element ─────────────
    if dry_run:
        microstates = parse_msf(msf_path)
        result      = {}
        rng         = np.random.default_rng(abs(hash(compound)) % 2**32)
        for ms in microstates:
            md      = MicrostateDescriptors(name=ms.name)
            elems   = _elements_in_xyz(ms.xyz_block)
            md.q_O  = float(rng.uniform(-0.8, -0.1)) if 'O' in elems else None
            md.q_N  = float(rng.uniform(-0.6,  0.1)) if 'N' in elems else None
            md.q_S  = float(rng.uniform(-0.5,  0.0)) if 'S' in elems else None
            md.q_C  = float(rng.uniform(-0.3,  0.3)) if 'C' in elems else None
            md.dipole  = float(rng.uniform(1.0, 8.0))
            md.homo_ev = float(rng.uniform(-12.0, -6.0))
            md.lumo_ev = float(rng.uniform(-2.0,  2.0))
            md.n_conf  = 1
            result[ms.name] = md
        save_descriptor_cache(msf_path, {n: _md_to_dict(md) for n, md in result.items()}, out_dir)
        return result

    # ── Determine which elements to track from MSF geometries ─────────────────
    microstates = parse_msf(msf_path)
    sorted_ms   = sorted(microstates, key=lambda m: m.n_protons, reverse=True)
    all_elems   = set()
    for ms in sorted_ms:
        all_elems |= _elements_in_xyz(ms.xyz_block)
    tracked = sorted(all_elems - {'H'})   # heavy atoms only, consistent order
    logging.info("  [%s] Tracking elements: %s", compound, tracked)

    # ── Extract per-conformer descriptors for each microstate ─────────────────
    result: dict[str, MicrostateDescriptors] = {}
    for ms in sorted_ms:
        ms_dir  = work_dir / ms.name
        md      = MicrostateDescriptors(name=ms.name)

        eq_list:    list[dict[str, float]] = []
        dip_list:   list[float]            = []
        homo_list:  list[float]            = []
        lumo_list:  list[float]            = []
        cpcm_list:  list[float]            = []
        weights:    list[float]            = []
        chelpg_list: list                  = []   # per-conformer CHELPG charge lists

        for struct_dir in sorted(ms_dir.glob("s*")):
            freq_outs = list(struct_dir.glob("*_freq.out"))
            dft_outs  = list(struct_dir.glob("*_dft.out"))
            if not freq_outs:
                continue
            freq_out = freq_outs[0]

            # Descriptors (charges, dipole, HOMO, LUMO, E_CPCM) come from _dft.out.
            # The GFN2-xTB _freq.out redirects all population analysis to a
            # sidecar 'properties.out' file — no charge block exists in _freq.out.
            # The DFT SP _dft.out always has a full MULLIKEN ATOMIC CHARGES block,
            # dipole moment, orbital energies table, and CPCM energy.
            if not dft_outs:
                logging.debug("  [%s/%s] No _dft.out in %s — skipping",
                              compound, ms.name, struct_dir.name)
                continue
            dft_out = dft_outs[0]

            eq, dip, homo, lumo, cpcm = _parse_descriptors_from_dft_out(dft_out, tracked)
            if eq is None or dip is None or homo is None:
                logging.debug("  [%s/%s] Incomplete descriptors in %s"
                              " (eq=%s dip=%s homo=%s)",
                              compound, ms.name, dft_out.name,
                              eq is not None, dip is not None, homo is not None)
                continue

            # Boltzmann weight from composite G — must resolve before appending
            # descriptors so all lists stay the same length as weights.
            try:
                G_xtb, E_xtb = _parse_gibbs_and_sp(freq_out)
                E_dft        = _parse_sp_energy(dft_out)
                weight = E_dft + (G_xtb - E_xtb)
            except (SCFDivergenceError, MissingThermoError):
                logging.debug("  [%s/%s] SCF/thermo failure in %s — skipping",
                              compound, ms.name, freq_out.name)
                continue
            except Exception:
                weight = 0.0

            # All checks passed — append atomically
            eq_list.append(eq)
            dip_list.append(dip)
            homo_list.append(homo)
            lumo_list.append(lumo if lumo is not None else homo)
            if cpcm is not None:
                cpcm_list.append(cpcm)
            weights.append(weight)
            # CHELPG per-atom charges for this conformer (None if absent)
            chelpg_list.append(_parse_chelpg_charges(dft_out.read_text(errors='replace')))

        if not eq_list:
            logging.warning("  [%s/%s] No descriptor data found in %s",
                            compound, ms.name, ms_dir)
            result[ms.name] = md
            continue

        # Boltzmann weighting over conformers
        if len(weights) > 1:
            w  = np.array(weights, dtype=float)
            w -= w.min()
            kT = KB_EV * TEMPERATURE / EH_TO_EV
            bw = np.exp(-w / kT)
            bw /= bw.sum()
        else:
            bw = np.array([1.0])

        # Weighted averages
        def wavg(vals):
            return float(np.dot(bw, vals))

        md.q_O      = wavg([eq.get('O', 0.0) for eq in eq_list]) if 'O' in tracked else None
        md.q_N      = wavg([eq.get('N', 0.0) for eq in eq_list]) if 'N' in tracked else None
        md.q_S      = wavg([eq.get('S', 0.0) for eq in eq_list]) if 'S' in tracked else None
        md.q_C      = wavg([eq.get('C', 0.0) for eq in eq_list]) if 'C' in tracked else None
        md.dipole   = wavg(dip_list)
        md.homo_ev  = wavg(homo_list)
        md.lumo_ev  = wavg(lumo_list)
        md.e_cpcm_eh = wavg(cpcm_list) if cpcm_list else None
        md.n_conf   = len(eq_list)

        # Mayer bond orders — best conformer only (atom-pair ordering is
        # conformer-dependent so Boltzmann averaging is not meaningful).
        best_idx = int(np.argmax(bw))
        all_dft_outs = sorted(ms_dir.glob("s*/*_dft.out"))
        if all_dft_outs and best_idx < len(all_dft_outs):
            best_txt = all_dft_outs[best_idx].read_text(errors='replace')
            md.mayer_bos = _parse_mayer_bond_orders(best_txt)

        # CHELPG charges — Boltzmann-averaged over all conformers.
        # Unlike Mayer BOs, CHELPG charges are indexed by atom number which
        # is invariant across conformers (ORCA preserves atom ordering from the
        # input xyz). Boltzmann averaging is therefore straightforward and
        # physically correct: for a flexible amine like tramadol or propranolol,
        # the ESP at the N atom varies measurably between conformers, and the
        # weighted average better represents the thermodynamic ensemble.
        valid_chelpg = [(c, w) for c, w in zip(chelpg_list, bw)
                        if c is not None and len(c) > 0]
        if valid_chelpg:
            n_atoms = len(valid_chelpg[0][0])
            # Check all conformers have the same atom count (they must)
            if all(len(c) == n_atoms for c, _ in valid_chelpg):
                w_sum = sum(w for _, w in valid_chelpg)
                avg = [sum(c[i] * w for c, w in valid_chelpg) / w_sum
                       for i in range(n_atoms)]
                md.chelpg_charges = avg

        result[ms.name] = md

    save_descriptor_cache(msf_path, {n: _md_to_dict(md) for n, md in result.items()}, out_dir)
    return result


def _md_to_dict(md: MicrostateDescriptors) -> dict:
    d = {
        "q_O":       md.q_O,
        "q_N":       md.q_N,
        "q_S":       md.q_S,
        "q_C":       md.q_C,
        "dipole":    md.dipole,
        "homo_ev":   md.homo_ev,
        "lumo_ev":   md.lumo_ev,
        "e_cpcm_eh": md.e_cpcm_eh,
        "n_conf":    md.n_conf,
    }
    # Mayer bond orders serialised as list of [i, j, bo] triples
    if md.mayer_bos is not None:
        d["mayer_bos"] = [[i, j, bo] for (i, j), bo in md.mayer_bos.items()]
    # CHELPG charges serialised as flat list indexed by atom number
    if md.chelpg_charges is not None:
        d["chelpg_charges"] = md.chelpg_charges
    return d

# ──────────────────────────────────────────────────────────────────────────────
# Build step descriptors from G_eff cache + descriptor cache
# ──────────────────────────────────────────────────────────────────────────────

def build_step_descriptors(
        msf_path:    Path,
        out_dir:     Path,
        temperature: float = TEMPERATURE,
        dry_run:     bool  = False,
        orca_binary: Optional[str] = None,
        nprocs:      int   = 56,
        method:      str   = "r2SCAN-3c",
        solvent_dft: str   = "CPCM(Water)",
) -> list[StepDescriptors]:
    """
    Combine G_eff and physical descriptors into StepDescriptors for
    each annotated ionisation step of one training compound.
    """
    RT_ln10     = KB_EV * temperature / EH_TO_EV * LN10
    compound    = msf_path.stem
    microstates = parse_msf(msf_path)
    sorted_ms   = sorted(microstates, key=lambda m: m.n_protons, reverse=True)

    # Check for required annotations
    usable = [(a, b) for a, b in zip(sorted_ms[:-1], sorted_ms[1:])
              if b.pka_step is not None]
    if not usable:
        return []

    # ── Thermo audit: check all freq.out files for missing Gibbs block ────────
    # Missing thermo introduces ~10–23 pKa unit errors in G_eff.  These cases
    # need to be re-run with pka_calibrate.py (which now retries with robust
    # SCF settings automatically).
    if not dry_run:
        work_dir = out_dir / f"work_{compound}"
        n_missing_thermo = 0
        for ms in sorted_ms:
            ms_dir = work_dir / ms.name
            for struct_dir in sorted(ms_dir.glob("s*")):
                for freq_out in struct_dir.glob("*_freq.out"):
                    try:
                        _parse_gibbs_and_sp(freq_out)
                    except MissingThermoError:
                        logging.warning(
                            "  [%s/%s] MISSING THERMO: %s has no Gibbs block. "
                            "Re-run pka_calibrate.py to trigger robust retry.",
                            compound, ms.name, freq_out.name,
                        )
                        n_missing_thermo += 1
                    except (SCFDivergenceError, RuntimeError):
                        pass   # these are handled elsewhere
        if n_missing_thermo:
            logging.warning(
                "  [%s] %d freq.out file(s) lack thermochemistry data. "
                "The G_eff for this compound may be unreliable. "
                "Re-run pka_calibrate.py to fix.",
                compound, n_missing_thermo,
            )

    # Load G_eff from cache
    g_cache = load_cached_G_eff(msf_path, out_dir)
    if g_cache is None:
        logging.warning("[%s] No G_eff cache — run pka_calibrate.py first.", compound)
        return []
    for ms in microstates:
        if ms.name in g_cache:
            ms.G_eff = g_cache[ms.name]

    # Before reading descriptors, check whether the DFT outputs were generated
    # with the current input template (i.e. include %chelpg block).  If any
    # conformer's DFT output predates the CHELPG addition, rerun that DFT SP
    # in-place — GOAT and xTB are untouched.
    if not dry_run and orca_binary:
        for ms in sorted_ms:
            ms_dir = out_dir / f"work_{compound}" / ms.name
            if ms_dir.exists():
                _rerun_dft_if_needed(ms, ms_dir, orca_binary, nprocs,
                                     method, solvent_dft)
    desc_map = extract_descriptors_for_compound(msf_path, out_dir, dry_run=dry_run)

    steps: list[StepDescriptors] = []
    for ms_acid, ms_base in usable:
        if ms_acid.G_eff is None or ms_base.G_eff is None:
            continue

        dG      = ms_base.G_eff - ms_acid.G_eff
        pka_c   = dG / RT_ln10

        # Sanity check: pKa_calc outside [0, 300] indicates a corrupted G_eff
        # (most likely from an imaginary-frequency conformer that slipped through
        # before the MissingThermoError fix, and is still in an old _geff.json
        # cache). Log an error and skip — do not include in descriptors or model.
        PKA_CALC_MIN, PKA_CALC_MAX = 0.0, 300.0
        if not (PKA_CALC_MIN <= pka_c <= PKA_CALC_MAX):
            logging.error(
                "  [%s] %s→%s: pKa_calc=%.2f is outside [%.0f, %.0f] — "
                "G_eff is corrupt (likely from an imaginary-frequency conformer). "
                "Delete %s_geff.json and %s_descriptors.json and rerun.",
                compound, ms_acid.name, ms_base.name, pka_c,
                PKA_CALC_MIN, PKA_CALC_MAX, compound, compound,
            )
            continue

        d_acid  = desc_map.get(ms_acid.name)
        d_base  = desc_map.get(ms_base.name)

        sd = StepDescriptors(
            compound         = compound,
            acid_name        = ms_acid.name,
            base_name        = ms_base.name,
            acid_charge      = ms_acid.charge,
            pka_calc         = pka_c,
            pka_exp          = ms_base.pka_step,
            functional_group = ms_base.functional_group,  # from MSF annotation
        )

        # Structural counts from acid xyz (zero extra cost)
        elem_counts = _count_elements(ms_acid.xyz_block)
        sd.n_heavy = sum(v for k, v in elem_counts.items() if k != 'H')
        sd.n_O     = elem_counts.get('O', 0)
        sd.n_N     = elem_counts.get('N', 0)
        sd.n_S     = elem_counts.get('S', 0)

        def _has_data(md: Optional[MicrostateDescriptors]) -> bool:
            """True only if md exists AND has at least one parseable descriptor."""
            return (md is not None
                    and any(v is not None for v in [md.q_O, md.q_N, md.q_S,
                                                    md.dipole, md.homo_ev]))

        if _has_data(d_acid) and _has_data(d_base):
            def _dq(a, b): return (b - a) if (a is not None and b is not None) else None
            sd.dq_O     = _dq(d_acid.q_O,      d_base.q_O)
            sd.dq_N     = _dq(d_acid.q_N,      d_base.q_N)
            sd.dq_S     = _dq(d_acid.q_S,      d_base.q_S)
            sd.dq_C     = _dq(d_acid.q_C,      d_base.q_C)
            sd.d_dipole = _dq(d_acid.dipole,    d_base.dipole)
            sd.d_homo   = _dq(d_acid.homo_ev,   d_base.homo_ev)
            sd.d_lumo   = _dq(d_acid.lumo_ev,   d_base.lumo_ev)
            sd.d_cpcm   = _dq(d_acid.e_cpcm_eh, d_base.e_cpcm_eh)
            # HOMO-LUMO gap = LUMO - HOMO; Δgap = gap(base) - gap(acid)
            if (d_acid.homo_ev is not None and d_acid.lumo_ev is not None and
                    d_base.homo_ev is not None and d_base.lumo_ev is not None):
                gap_acid  = d_acid.lumo_ev - d_acid.homo_ev
                gap_base  = d_base.lumo_ev - d_base.homo_ev
                sd.d_gap  = gap_base - gap_acid

            # Mayer bond order descriptors
            if d_acid.mayer_bos is not None and d_base.mayer_bos is not None:
                sd.dbo_xc, sd.dbo_max, sd.dbo_ring = _bond_order_descriptors(
                    ms_acid.xyz_block, ms_base.xyz_block,
                    d_acid.mayer_bos,  d_base.mayer_bos,
                )

            # CHELPG electrostatic potential descriptors
            if d_acid.chelpg_charges is not None or d_base.chelpg_charges is not None:
                sd.q_chelpg_site_acid, sd.dq_chelpg_site = _chelpg_descriptors(
                    ms_acid.xyz_block, ms_base.xyz_block,
                    d_acid.chelpg_charges, d_base.chelpg_charges,
                )

        def _fmt(v): return f"{v:+.3f}" if v is not None else "n/a"
        logging.info(
            "  [%s] %s→%s  q=%+d  ΔG_DFT=%+.2f kcal/mol  "
            "Δq_O=%s  Δq_N=%s  Δq_S=%s  Δq_C=%s  Δ|μ|=%s  "
            "ΔE_HOMO=%s  ΔE_LUMO=%s  ΔE_CPCM=%s",
            compound, ms_acid.name, ms_base.name, ms_acid.charge,
            dG * EH_TO_KCAL,
            _fmt(sd.dq_O), _fmt(sd.dq_N), _fmt(sd.dq_S), _fmt(sd.dq_C),
            _fmt(sd.d_dipole), _fmt(sd.d_homo), _fmt(sd.d_lumo), _fmt(sd.d_cpcm),
        )
        steps.append(sd)

    # Compute ECFP4 fingerprint once per compound and assign to all its steps.
    # All steps for a compound share the same molecular scaffold fingerprint —
    # the Tanimoto kernel measures compound identity, not protonation state.
    fp = _compute_ecfp4(msf_path)
    fp_fallback = np.zeros(1024, dtype=np.float32)
    for sd in steps:
        sd.fp_bits = fp if fp is not None else fp_fallback

    return steps

# ──────────────────────────────────────────────────────────────────────────────
# GPR model
# ──────────────────────────────────────────────────────────────────────────────

class GPRModel:
    """
    Gaussian Process Regression for pKa correction.

    Kernel: Matérn ν=5/2 with per-feature ARD length scales.

    Feature vector (6-dim, v6):
        [δpKa_calc, q_acid, Δq_O, Δq_N, Δq_C, Σ|Δq|]

    All 6 features are active (ls < 10.0) at N=86 with bound=10.0.
    Orbital/dipole features (ΔE_LUMO, ΔE_HOMO, ΔE_gap, Δ|μ|) are wall-bound
    (ls=10.0) because their partial correlation with the correction, given
    δpKa_calc, is near zero — both encode deprotonation energy. Confirmed by
    simulation: increasing ls_max beyond 10.0 yields k(1σ) > 0.9999, which
    is computationally indistinguishable from amplitude rescaling.

    Feature vector (12-dim):
        [δpKa_calc, q_acid, Δq_C, ΔE_CPCM, f_O, f_N, f_S, Σ|Δq|/n_hvy, ΔE_LUMO,
         ΔBO_XC, ΔBO_max, ΔBO_ring]

    ΔBO features (Mayer bond orders) are 0.0 for existing training compounds
    (outputs without Print[P_BondOrders] 1) and real values for all new
    compounds going forward. They activate as the training set grows.

    N/D = 200/6 = 33.3.
    """

    FEATURE_NAMES  = ["δpKa_calc", "q_acid", "ΔE_CPCM",
                       "ΔBO_XC", "ΔBO_ring", "q_CHELPG_site"]
    FALLBACK_NAMES = ["δpKa_calc", "q_acid"]

    def __init__(self):
        self.alpha_noise   = 0.5
        self.length_scales = None
        self.amplitude     = None
        self.X_train       = None
        self.y_train       = None
        self.X_mean        = None
        self.X_std         = None
        self.K_inv         = None
        self.n_train       = 0
        self.n_full        = 0
        self.n_partial     = 0
        self.rmse_loo      = None
        self.mae_loo       = None
        self.feature_mode  = "full"
        self.charge_means: dict[int, float] = {}
        self.trend_coeffs: dict[int, tuple[float,float]] = {}
        self.fps_train:    Optional[np.ndarray] = None   # (N, B) ECFP4 fingerprints
        self.K_T:          Optional[np.ndarray] = None   # (N, N) Tanimoto matrix
        self._weights: Optional[np.ndarray] = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Matérn ν=5/2 kernel with per-feature ARD length scales.

        k(r) = amp² · (1 + √5·r + 5r²/3) · exp(-√5·r)
        where r = sqrt(Σ_d (x1_d - x2_d)² / ls_d²)

        Matérn ν=5/2 is preferred over RBF for pKa prediction:
        - pKa surfaces are twice-differentiable but NOT infinitely smooth
          across functional group boundaries (carboxylate vs phenol, etc.)
        - RBF assumes infinite differentiability → oversmooths at boundaries
        - Matérn ν=5/2 decays faster at large r → lower similarity for
          chemically dissimilar compounds → benchmark shows ~10% RMSE gain

        Benchmark (N=59, LOO):
            RBF:      RMSE=1.833
            M ν=5/2:  RMSE~1.71 (10-fold CV; ~1.75-1.80 expected LOO)
        """
        # Scaled squared distance per feature: (N1, N2, D)
        diff = (X1[:, None, :] - X2[None, :, :]) / self.length_scales
        r2   = np.sum(diff ** 2, axis=-1)          # (N1, N2)
        r    = np.sqrt(np.maximum(r2, 0.0))         # numerical safety
        sqrt5_r = np.sqrt(5.0) * r
        return self.amplitude ** 2 * (1.0 + sqrt5_r + sqrt5_r**2 / 3.0) * np.exp(-sqrt5_r)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.X_mean) / (self.X_std + 1e-12)

    def _nlml_and_grad(
            self,
            log_theta: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            K_T: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray]:
        """
        Negative log marginal likelihood and analytic gradient.

        When K_T (Tanimoto matrix) is provided, the kernel is a product:
            k_total(i,j) = k_Matérn(x_i, x_j) × T(fp_i, fp_j)

        Gradient of the product kernel:
            ∂k_total/∂θ = (∂k_Matérn/∂θ) ⊙ K_T
        where ⊙ is elementwise multiplication.  K_T has no trainable
        parameters — it is fixed at training time from the compound fingerprints.
        """
        D     = X.shape[1]
        amp   = float(np.exp(log_theta[0]))
        noise = float(np.exp(log_theta[1]))
        ls    = np.exp(log_theta[2:2 + D])

        self.amplitude, self.alpha_noise, self.length_scales = amp, noise, ls

        K_matern = self._kernel(X, X)
        K  = K_matern * K_T if K_T is not None else K_matern
        Kn = K + noise ** 2 * np.eye(len(y))

        try:
            L = np.linalg.cholesky(Kn)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(log_theta)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))   # Kn⁻¹ y
        Ki    = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(y))))  # Kn⁻¹

        nlml = (0.5 * float(y @ alpha)
                + np.sum(np.log(np.diag(L)))
                + 0.5 * len(y) * np.log(2 * np.pi))

        W = np.outer(alpha, alpha) - Ki       # (αα^T - Kn^{-1}): shared factor

        grad = np.zeros(2 + D)

        # d(NLML)/d log(amp): K_total = K_matern * K_T, so d(K_total)/d(amp) = 2*K_total/amp
        grad[0] = -0.5 * float(np.trace(W @ (2.0 * K)))
        # d(NLML)/d log(noise)
        grad[1] = -0.5 * float(np.trace(W) * 2.0 * noise ** 2)

        # d(NLML)/d log(ls_d): product rule gives dK_total/d_ls_d = dK_matern/dls_d * K_T
        diff3   = X[:, None, :] - X[None, :, :]
        r2_mat  = np.sum((diff3 / ls) ** 2, axis=-1)
        sqrt5_r = np.sqrt(5.0) * np.sqrt(np.maximum(r2_mat, 0.0))
        poly    = np.maximum(1.0 + sqrt5_r + sqrt5_r**2 / 3.0, 1e-12)
        factor  = 5.0 * (1.0 + sqrt5_r) / (3.0 * poly)
        for d in range(D):
            dK_matern_d = K_matern * factor * diff3[:, :, d] ** 2 / (ls[d] ** 2)
            dK_d        = dK_matern_d * K_T if K_T is not None else dK_matern_d
            grad[2 + d] = -0.5 * float(np.trace(W @ dK_d))

        return float(nlml), grad

    def fit(self, steps: list[StepDescriptors]) -> None:
        """
        Fit GPR via maximum marginal likelihood.

        All hyperparameters — amplitude, noise, and one length scale per
        feature — are optimised jointly using L-BFGS-B with analytic
        gradients until convergence (ftol=1e-12, gtol=1e-8).

        Multiple restarts from a log-uniform grid of starting points guard
        against local minima.  The best converged solution (lowest NLML) is
        kept.

        LOO predictions are computed analytically in O(N³) using the
        identity  μ_{-i} = y_i − (Kn⁻¹y)_i / (Kn⁻¹)_{ii}, avoiding N
        separate matrix inversions.
        """
        from scipy.optimize import minimize

        # ── Compute per-charge-state pKa_calc means (for centring) ───────────
        # δpKa_calc = pKa_calc - mean(pKa_calc | acid_charge) removes the
        # dominant ~200-unit DFT offset within each charge class so the kernel
        # length scale for the first feature becomes meaningful.
        charge_groups: dict[int, list[float]] = {}
        for s in steps:
            charge_groups.setdefault(s.acid_charge, []).append(s.pka_calc)
        self.charge_means = {q: float(np.mean(vals))
                             for q, vals in charge_groups.items()}
        logging.info("GPR: charge-state pKa_calc means: %s",
                     {q: f"{m:.1f}" for q, m in sorted(self.charge_means.items())})

        # ── Feature selection ─────────────────────────────────────────────────
        full    = [s for s in steps if s.feature_vector(self.charge_means) is not None]
        partial = [s for s in steps if s.feature_vector(self.charge_means) is None]
        self.n_full    = len(full)
        self.n_partial = len(partial)

        if len(full) >= 5:
            X_raw = np.array([s.feature_vector(self.charge_means) for s in full])
            y     = np.array([s.pka_exp for s in full])
            self.feature_mode = "full"
            logging.info("GPR: %d full-descriptor steps (%d partial skipped).",
                         len(full), len(partial))
            if partial:
                logging.warning("  %d steps missing full descriptors — "
                                "run pka_calibrate.py to generate ORCA outputs.",
                                len(partial))
        else:
            all_s = full + partial
            X_raw = np.array([s.feature_vector_partial(self.charge_means) for s in all_s])
            y     = np.array([s.pka_exp for s in all_s])
            self.feature_mode = "partial"
            logging.warning("GPR: only %d full-descriptor steps — "
                            "falling back to [δpKa_calc, q_acid] features.",
                            len(full))

        # ── Standardise features ──────────────────────────────────────────────
        self.X_mean = X_raw.mean(axis=0)
        self.X_std  = X_raw.std(axis=0).clip(min=1e-8)
        X = (X_raw - self.X_mean) / self.X_std
        D, N = X.shape[1], len(y)
        self.n_train = N

        # ── Fit linear mean function μ(x) = α·δpKa_calc + β per charge state ─
        # The prior mean is a per-charge-state OLS fit on the first feature
        # (δpKa_calc), which is the dominant linear driver by construction.
        #
        # Motivation: a constant-mean GPR forces the kernel to model the full
        # dynamic range of pKa (0–14). At the extremes of the pKa_calc
        # distribution, kernel correlation weakens and predictions revert to
        # the training mean — producing the sigmoidal residual pattern observed
        # in the Hammett/Taft test set (RMSE=1.03, but σ≈2.0 at low pKa_calc).
        #
        # With a linear mean the GP only models the RESIDUAL from the LFER
        # trend. Since pKa_exp ≈ α·δpKa_calc + β by the Hammett equation, the
        # residuals are small (≲ 2 pKa units) and stationary — the kernel
        # interpolates them far more reliably with fewer training points.
        #
        # Implementation: one OLS regression (α, β) per charge state on the
        # raw (unstandardised) first feature δpKa_calc and pKa_exp. The GPR
        # then fits y_res = pKa_exp − μ(x), and predict() adds μ(x_*) back.
        #
        # A separate slope per charge state is necessary because the LFER
        # slope differs systematically: carboxylates (q=0) have a shallower
        # slope than ammonium groups (q=+1) which tend to have a steeper slope
        # due to larger DFT solvation errors for cationic species.
        self.trend_coeffs: dict[int, tuple[float, float]] = {}
        steps_used = full if self.feature_mode == "full" else full + partial
        charge_groups_fit: dict[int, list] = {}
        for s, xi in zip(steps_used, X_raw):
            charge_groups_fit.setdefault(s.acid_charge, []).append(
                (xi[0], s.pka_exp)  # xi[0] = δpKa_calc (standardised not yet applied)
            )

        # Use raw (non-standardised) δpKa_calc for the trend — keeps the
        # slope physically interpretable (pKa_exp per pKa_calc unit)
        dpka_raw = X_raw[:, 0]   # first column = δpKa_calc before standardisation
        y_trend  = np.zeros(N)
        for q, pairs in charge_groups_fit.items():
            xs = np.array([p[0] for p in pairs])
            ys = np.array([p[1] for p in pairs])
            idx = np.array([i for i, s in enumerate(steps_used)
                            if s.acid_charge == q])
            if len(xs) >= 2:
                # OLS: [1, δpKa_calc] @ [β, α] = pKa_exp
                A = np.column_stack([np.ones_like(xs), xs])
                coeffs, *_ = np.linalg.lstsq(A, ys, rcond=None)
                beta, alpha = float(coeffs[0]), float(coeffs[1])
            else:
                # Only 1 point: use mean as intercept, zero slope
                alpha, beta = 0.0, float(ys[0])
            self.trend_coeffs[q] = (alpha, beta)
            y_trend[idx] = alpha * dpka_raw[idx] + beta

        y_res = y - y_trend   # residuals: what the GP kernel needs to model
        sig_y = float(np.std(y_res))  # noise scale from residuals, not raw y

        # Log the fitted linear trend
        for q in sorted(self.trend_coeffs):
            a, b = self.trend_coeffs[q]
            logging.info("GPR linear trend q=%+d: pKa = %.4f·δpKa_calc + %.4f",
                         q, a, b)

        # ── Hyperparameter bounds in log space ────────────────────────────────
        # amp   ∈ [0.1·σ_y,  10·σ_y]
        # noise ∈ [0.30,      5.0]   pKa units
        #   Lower bound of 0.30 is a hard physics floor:
        #   - Experimental pKa uncertainty: ~0.05 units
        #   - DFT/CPCM intrinsic error:     ~0.5 units
        #   - Together these set a minimum noise ~0.3 pKa units.
        #   Allowing noise < 0.3 lets the GPR interpolate through training
        #   points exactly (training error → 0) while LOO error stays large —
        #   the classic overfitting signature seen when N/D < 5.
        # ls_d  ∈ [0.1, 10.0]  (in standardised feature space)
        #   Upper bound restored to 10.0 (was 5.0).
        #   Rationale: at bound=5.0, features with natural optimal ls in the
        #   3–5 range (Δq_N≈4.2, Δq_C≈3.7, Σ|Δq|≈2.9 at N=82) were clipped
        #   at the wall, appearing inactive while actually providing information.
        #   Truly inactive features (ΔE_HOMO, ΔE_gap, Δ|μ|) will hit 10.0
        #   instead — a negligible kernel contribution, not a hard cutoff.
        #   The report threshold for "ACTIVE" is set at ls < 9.5 to distinguish
        #   naturally converged features from wall-bound ones.
        NOISE_MIN = 0.30
        LS_MAX    = 10.0
        bounds = (
            [(np.log(0.1 * sig_y), np.log(10.0 * sig_y))]
            + [(np.log(NOISE_MIN), np.log(5.0))]
            + [(np.log(0.1), np.log(LS_MAX))] * D
        )

        # ── Tanimoto product kernel matrix ────────────────────────────────────
        # Precompute the (N, N) Tanimoto matrix from ECFP4 fingerprints.
        # K_total = K_Matérn ⊙ K_T  (elementwise product)
        # K_T is fixed throughout optimisation (no trainable parameters).
        # Compounds with failed fingerprint generation have fp=zeros → T=1.0.
        all_fps  = [s.fp_bits if s.fp_bits is not None
                    else np.zeros(1024, dtype=np.float32)
                    for s in steps_used]
        fps_mat  = np.array(all_fps, dtype=np.float64)          # (N, B)
        K_T      = _tanimoto_matrix(fps_mat)                     # (N, N)
        self.fps_train = fps_mat
        self.K_T       = K_T

        n_tanimoto_active = int((K_T < 0.999).sum() // 2)
        logging.info("GPR: Tanimoto product kernel — %d compound-pair similarities "
                     "< 1.0 (mean off-diagonal T=%.3f)",
                     n_tanimoto_active,
                     float(K_T[K_T < 1.0].mean()) if n_tanimoto_active else 1.0)

        # ── Multi-start L-BFGS-B ─────────────────────────────────────────────
        # Grid: 3 values each for amp, noise, ls = 3³ = 27 restarts.
        starts = [
            (amp0, noise0, ls0)
            for amp0   in [0.5 * sig_y, sig_y, 2.0 * sig_y]
            for noise0 in [max(NOISE_MIN, 0.1), max(NOISE_MIN, 0.4), max(NOISE_MIN, 1.0)]
            for ls0    in [0.3, 1.0, 3.0]
        ]

        best_nlml  = np.inf
        best_theta = None

        for amp0, noise0, ls0 in starts:
            theta0 = np.array(
                [np.log(amp0), np.log(noise0)] + [np.log(ls0)] * D
            ).clip(
                [b[0] for b in bounds],
                [b[1] for b in bounds],
            )
            res = minimize(
                fun     = self._nlml_and_grad,
                x0      = theta0,
                args    = (X, y_res, K_T),
                method  = "L-BFGS-B",
                jac     = True,
                bounds  = bounds,
                options = {"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.fun < best_nlml:
                best_nlml  = res.fun
                best_theta = res.x.copy()

        # Apply best converged hyperparameters
        self.amplitude     = float(np.exp(best_theta[0]))
        self.alpha_noise   = float(np.exp(best_theta[1]))
        self.length_scales = np.exp(best_theta[2:2 + D])

        feat_names = (self.FEATURE_NAMES if self.feature_mode == "full"
                      else self.FALLBACK_NAMES)
        ls_str = "  ".join(f"{n}={ls:.3f}"
                           for n, ls in zip(feat_names, self.length_scales))
        logging.info(
            "GPR converged (NLML=%.4f, %d restarts):\n"
            "  amplitude=%.4f  noise=%.4f\n"
            "  length_scales: %s",
            best_nlml, len(starts),
            self.amplitude, self.alpha_noise, ls_str,
        )

        # ── Final Kn and inverse (product kernel: Matérn × Tanimoto) ─────────
        K  = self._kernel(X, X) * K_T   # product: Matérn × Tanimoto
        Kn = K + self.alpha_noise ** 2 * np.eye(N)
        self.K_inv   = np.linalg.inv(Kn)
        self.X_train = X
        self.y_train = y_res   # residuals after linear trend

        # ── Analytic LOO in O(N³) ─────────────────────────────────────────────
        # From Rasmussen & Williams (5.12):
        #   μ_{-i} = y_i − (Kn⁻¹ y)_i / (Kn⁻¹)_{ii}
        # Applied to residuals; trend is added back to compare with pKa_exp.
        alpha_vec  = self.K_inv @ y_res
        kii_diag   = np.diag(self.K_inv)
        loo_res    = y_res - alpha_vec / kii_diag   # LOO residual predictions
        loo_preds  = loo_res + y_trend              # add trend back → pKa scale
        residuals  = loo_preds - y                  # error vs experimental pKa
        self.rmse_loo = float(np.sqrt(np.mean(residuals ** 2)))
        self.mae_loo  = float(np.mean(np.abs(residuals)))
        logging.info("GPR LOO  RMSE=%.3f  MAE=%.3f pKa units",
                     self.rmse_loo, self.mae_loo)


    def predict(self, step: StepDescriptors) -> tuple[float, float]:
        """
        Predict pKa and uncertainty (σ) for one step.
        Returns (pKa_pred, sigma).
        """
        if self.feature_mode == "full":
            x_raw = step.feature_vector(self.charge_means)
            if x_raw is None:
                x_raw = np.concatenate([
                    step.feature_vector_partial(self.charge_means),
                    np.zeros(len(self.FEATURE_NAMES) - 2),
                ])
                logging.warning("Prediction for %s/%s uses zero-padded descriptors.",
                                 step.compound, step.acid_name)
        else:
            x_raw = step.feature_vector_partial(self.charge_means)

        # Guard against feature-vector / model dimension mismatch.
        # This happens when the model was trained with D features but the
        # current FEATURE_NAMES list has changed (e.g. ΔE_CPCM was added).
        model_D  = len(self.X_mean)
        vector_D = len(x_raw)
        if vector_D != model_D:
            raise ValueError(
                f"Feature dimension mismatch: model expects D={model_D} "
                f"({list(self.X_mean.shape)}), but feature_vector() returned "
                f"D={vector_D} ({StepDescriptors.FEATURE_NAMES}).\n"
                f"The model on disk was trained with a different feature set.\n"
                f"Fix: delete the stale caches and retrain:\n"
                f"  rm pka_results/*_descriptors.json\n"
                f"  python pka_gpr_calibrate.py training_set/*.msf --train "
                f"--outdir pka_results --orca <orca> --nprocs <N>"
            )

        x  = self._scale(x_raw.reshape(1, -1))
        k_matern = self._kernel(self.X_train, x).flatten()

        # Apply Tanimoto product kernel: k_total = k_Matérn ⊙ k_T_star
        if self.fps_train is not None:
            fp_test = (step.fp_bits if step.fp_bits is not None
                       else np.zeros(1024, dtype=np.float32)).astype(np.float64)
            k_T_star = _tanimoto_vector(fp_test, self.fps_train)
            k_ = k_matern * k_T_star
        else:
            k_ = k_matern

        # Kernel prediction is of the residual; add linear trend back
        mu_res = float(k_ @ self.K_inv @ self.y_train)
        q  = step.acid_charge
        alpha_t, beta_t = self.trend_coeffs.get(q, (0.0, float(np.mean(self.y_train))))
        dpka_raw = x_raw[0]
        mu_trend = alpha_t * dpka_raw + beta_t
        mu  = mu_res + mu_trend

        var = float(self.amplitude ** 2 - k_ @ self.K_inv @ k_)
        sigma = float(np.sqrt(max(var, 0.0)) + self.alpha_noise)
        return mu, sigma

    def to_dict(self) -> dict:
        return {
            "feature_mode":   self.feature_mode,
            "feature_names":  self.FEATURE_NAMES if self.feature_mode == "full"
                              else self.FALLBACK_NAMES,
            "charge_means":   {str(k): v for k, v in self.charge_means.items()},
            "trend_coeffs":   {str(k): list(v)
                               for k, v in self.trend_coeffs.items()},
            "weights":        self._weights.tolist() if self._weights is not None else None,
            "n_train":        self.n_train,
            "n_full":         self.n_full,
            "n_partial":      self.n_partial,
            "amplitude":      self.amplitude,
            "alpha_noise":    self.alpha_noise,
            "length_scales":  self.length_scales.tolist(),
            "X_mean":         self.X_mean.tolist(),
            "X_std":          self.X_std.tolist(),
            "X_train":        self.X_train.tolist(),
            "y_train":        self.y_train.tolist(),   # residuals after trend
            "K_inv":          self.K_inv.tolist(),
            "fps_train":      self.fps_train.tolist() if self.fps_train is not None else None,
            "rmse_loo":       self.rmse_loo,
            "mae_loo":        self.mae_loo,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GPRModel":
        m = cls()
        m.feature_mode   = d["feature_mode"]
        m.n_train        = d["n_train"]
        m.n_full         = d.get("n_full", d["n_train"])
        m.n_partial      = d.get("n_partial", 0)
        m.amplitude      = d["amplitude"]
        m.alpha_noise    = d["alpha_noise"]
        m.length_scales  = np.array(d["length_scales"])
        m.X_mean         = np.array(d["X_mean"])
        m.X_std          = np.array(d["X_std"])
        m.X_train        = np.array(d["X_train"])
        m.y_train        = np.array(d["y_train"])
        m.K_inv          = np.array(d["K_inv"])
        fps_raw = d.get("fps_train")
        m.fps_train = np.array(fps_raw, dtype=np.float64) if fps_raw is not None else None
        m.rmse_loo       = d.get("rmse_loo")
        m.mae_loo        = d.get("mae_loo")
        m.charge_means   = {int(k): float(v)
                            for k, v in d.get("charge_means", {}).items()}
        # Linear trend coefficients: {charge: (alpha, beta)}
        # Fall back to zero slope / zero intercept for models saved before
        # the linear mean was added (graceful backward compatibility).
        raw_tc = d.get("trend_coeffs", {})
        m.trend_coeffs   = {int(k): (float(v[0]), float(v[1]))
                            for k, v in raw_tc.items()}
        w = d.get("weights")
        m._weights       = np.array(w) if w is not None else None
        return m

# ──────────────────────────────────────────────────────────────────────────────
# Save / load GPR model
# ──────────────────────────────────────────────────────────────────────────────

def save_gpr_model(model: GPRModel, steps: list[StepDescriptors],
                   method: str, out_path: Path) -> None:
    data = {
        "method":  method,
        "model":   model.to_dict(),
        "training_steps": [
            {
                "compound":          s.compound,
                "acid_name":         s.acid_name,
                "base_name":         s.base_name,
                "acid_charge":       s.acid_charge,
                "pka_calc":          s.pka_calc,
                "pka_exp":           s.pka_exp,
                "functional_group":  s.functional_group,
                "dq_O":              s.dq_O,
                "dq_N":              s.dq_N,
                "dq_S":              s.dq_S,
                "dq_C":              s.dq_C,
                "d_dipole":          s.d_dipole,
                "d_homo":            s.d_homo,
                "d_lumo":            s.d_lumo,
                "d_gap":             s.d_gap,
                "d_cpcm":            s.d_cpcm,
                "dbo_xc":            s.dbo_xc,
                "dbo_max":           s.dbo_max,
                "dbo_ring":          s.dbo_ring,
                "q_chelpg_site_acid":s.q_chelpg_site_acid,
                "dq_chelpg_site":    s.dq_chelpg_site,
                "n_heavy":           s.n_heavy,
                "n_O":               s.n_O,
                "n_N":               s.n_N,
                "n_S":               s.n_S,
            }
            for s in steps
        ],
    }
    out_path.write_text(json.dumps(data, indent=2))
    logging.info("GPR model saved → %s", out_path)


def load_gpr_model(json_path: Path) -> tuple[GPRModel, str]:
    """Returns (model, method_str)."""
    data  = json.loads(json_path.read_text())
    model = GPRModel.from_dict(data["model"])
    return model, data.get("method", "unknown")

# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def write_gpr_report(model: GPRModel, steps: list[StepDescriptors],
                     method: str, out_path: Path) -> None:
    lines = [
        "=" * 72,
        "  pKa GPR Calibration Report",
        f"  DFT method: {method} / CPCM(Water) + GFN2-xTB RRHO",
        "=" * 72,
        "",
        "  Model: Gaussian Process Regression",
        f"  Feature mode:  {model.feature_mode}",
        f"  Features:      {model.FEATURE_NAMES if model.feature_mode == 'full' else model.FALLBACK_NAMES}",
        "",
        f"  Training points:  {model.n_train} total",
        f"    Full descriptors: {model.n_full}",
        f"    Partial only:     {model.n_partial}",
        "",
        f"  Kernel amplitude:  {model.amplitude:.4f} pKa",
        f"  Noise level:       {model.alpha_noise:.4f} pKa",
        f"  Length scales:     {[f'{ls:.3f}' for ls in model.length_scales]}",
        "",
        f"  LOO RMSE:  {model.rmse_loo:.4f} pKa units",
        f"  LOO MAE:   {model.mae_loo:.4f} pKa units",
        "",
        "  Physical interpretation of descriptors:",
    ]

    # Dynamic per-feature interpretation — based on actual length scales
    # so the report stays accurate as the feature vector evolves.
    ls_bound = 9.5  # features at or above this are wall-bound (ls_max=10.0)
    feat_names = (model.FEATURE_NAMES if model.feature_mode == "full"
                  else model.FALLBACK_NAMES)
    descriptions = {
        "δpKa_calc":  "DFT deprotonation energy (charge-state centred); primary driver",
        "q_acid":     "Born solvation q² baseline; separates charge-state families",
        "Δq_O":       "O-centred charge shift; carboxylate/phenol/phosphate identity",
        "Δq_N":       "N-centred charge shift; ammonium/imidazolium/guanidinium",
        "Δq_S":       "S-centred charge shift; unique thiol identifier",
        "Δq_C":       "carbon π-redistribution; aromaticity and conjugation proxy",
        "Σ|Δq|":      "total charge reorganisation magnitude",
        "is_O_dom":   "FG indicator: O-dominant deprotonation (carboxylate/phenol)",
        "is_N_dom":   "FG indicator: N-dominant deprotonation (ammonium/imidazolium)",
        "is_S_dom":   "FG indicator: S-dominant deprotonation (thiol)",
        "ΔE_LUMO":    "LUMO energy shift; electron affinity of base; r=+0.51 with corr",
        "ΔE_HOMO":    "HOMO energy shift; proton affinity proxy",
        "ΔE_gap":     "HOMO-LUMO gap change; chemical hardness / aromaticity",
        "Δ|μ|":       "dipole moment change; spatial charge redistribution",
    }
    for i, (name, ls) in enumerate(zip(feat_names, model.length_scales)):
        status = "ACTIVE" if ls < ls_bound else f"wall-bound (ls={ls:.2f}≥{ls_bound}, max=10.0)"
        desc = descriptions.get(name, "")
        lines.append(f"    {name:12s}  ls={ls:5.3f}  {status:28s}  {desc}")

    # Count active features and flag inactive ones
    n_active = sum(1 for ls in model.length_scales if ls < ls_bound)
    n_total  = len(feat_names)
    lines += [
        "",
        f"  Active features: {n_active}/{n_total} "
        f"(length scale < {ls_bound}; wall-bound features ls→10.0 are negligible)",
        "",
        "─" * 72,
        "  Training data with predictions:",
        "",
        f"  {'Compound':>20}  {'Step':>20}  {'q':>3}  {'pKa_calc':>9}  "
        f"{'pKa_exp':>8}  {'pKa_pred':>9}  {'σ':>5}  {'err':>7}",
        "  " + "─" * 84,
    ]

    for s in steps:
        pred, sigma = model.predict(s)
        err  = pred - s.pka_exp
        step_label = f"{s.acid_name[:10]}→{s.base_name[:9]}"
        lines.append(
            f"  {s.compound:>20}  {step_label:>20}  {s.acid_charge:>+3d}  "
            f"{s.pka_calc:9.2f}  {s.pka_exp:8.3f}  {pred:9.3f}  "
            f"{sigma:5.3f}  {err:+7.3f}"
        )

    lines += [
        "",
        "─" * 72,
        "  Descriptor correlations with correction (pKa_exp - pKa_calc):",
        "",
    ]
    corrections = np.array([s.pka_exp - s.pka_calc for s in steps])
    for name, vals in [
        ("q_acid",   np.array([float(s.acid_charge)    for s in steps])),
        ("Δq_O",     np.array([s.dq_O    or 0.0        for s in steps])),
        ("Δq_N",     np.array([s.dq_N    or 0.0        for s in steps])),
        ("Δq_S",     np.array([s.dq_S    or 0.0        for s in steps])),
        ("Δq_C",     np.array([s.dq_C    or 0.0        for s in steps])),
        ("ΔE_LUMO",  np.array([s.d_lumo  or 0.0        for s in steps])),
        ("Σ|Δq|",    np.array([s._sum_abs_dq()          for s in steps])),
        ("n_heavy",  np.array([float(s.n_heavy)         for s in steps])),
        ("n_O",      np.array([float(s.n_O)             for s in steps])),
        ("n_N",      np.array([float(s.n_N)             for s in steps])),
    ]:
        if np.std(vals) > 0 and np.std(corrections) > 0:
            r = float(np.corrcoef(vals, corrections)[0, 1])
            lines.append(f"    {name:12s}:  r = {r:+.4f}")

    lines += ["", ""]
    out_path.write_text("\n".join(lines))
    logging.info("GPR report saved → %s", out_path)


def plot_gpr(model: GPRModel, steps: list[StepDescriptors],
             out_path: Path, method: str) -> None:

    def _dominant_element(s: StepDescriptors) -> str:
        """
        Infer the dominant ionising element from the Δq values.
        Whichever element has the largest |Δq| is the primary site.
        Returns 'O', 'N', 'S', or '?' when descriptors are absent.
        """
        candidates = {
            'O': abs(s.dq_O or 0.0),
            'N': abs(s.dq_N or 0.0),
            'S': abs(s.dq_S or 0.0),
        }
        # All zero → missing descriptors
        if max(candidates.values()) < 1e-6:
            return '?'
        return max(candidates, key=candidates.get)

    elem_colors = {'O': '#e74c3c', 'N': '#2980b9', 'S': '#f39c12', '?': '#7f8c8d'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Panel 1: pKa_pred vs pKa_exp ─────────────────────────────────────────
    ax     = axes[0]
    preds  = np.array([model.predict(s)[0] for s in steps])
    sigmas = np.array([model.predict(s)[1] for s in steps])
    y_exp  = np.array([s.pka_exp for s in steps])

    for s, pred, sigma in zip(steps, preds, sigmas):
        col = elem_colors[_dominant_element(s)]
        ax.errorbar(s.pka_exp, pred, yerr=sigma,
                    fmt='o', color=col, ecolor=col, elinewidth=1.2,
                    capsize=3, markersize=7, alpha=0.85, zorder=3)
        ax.annotate(s.compound, (s.pka_exp, pred),
                    fontsize=5.5, textcoords="offset points",
                    xytext=(4, 2), color="dimgrey")

    lo = min(y_exp.min(), preds.min()) - 0.5
    hi = max(y_exp.max(), preds.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="ideal")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("pKa_exp", fontsize=11)
    ax.set_ylabel("pKa_pred ± σ", fontsize=11)
    ax.set_title(f"GPR predictions\nLOO RMSE={model.rmse_loo:.3f}  MAE={model.mae_loo:.3f}",
                 fontsize=10)
    ax.grid(alpha=0.25)

    for sym, col in elem_colors.items():
        label = {'O': 'dominant O (carboxylate/phenol)',
                 'N': 'dominant N (ammonium/imidazolium)',
                 'S': 'dominant S (thiol)',
                 '?': 'unknown'}.get(sym, sym)
        ax.scatter([], [], color=col, label=label, s=50)
    ax.legend(fontsize=7, loc="upper left")

    # ── Panel 2: residuals vs pKa_calc, coloured by acid charge ──────────────
    ax2       = axes[1]
    residuals = preds - y_exp
    pka_calcs = np.array([s.pka_calc for s in steps])
    charges   = np.array([s.acid_charge for s in steps])
    unique_q  = sorted(set(int(q) for q in charges))
    q_cmap    = plt.cm.RdYlBu(np.linspace(0.1, 0.9, max(len(unique_q), 1)))
    q_colors  = {q: q_cmap[i] for i, q in enumerate(unique_q)}

    for s, calc, res in zip(steps, pka_calcs, residuals):
        ax2.scatter(calc, res, color=q_colors[s.acid_charge],
                    s=60, zorder=3, alpha=0.85)
        ax2.annotate(s.compound, (calc, res),
                     fontsize=5.5, textcoords="offset points",
                     xytext=(4, 2), color="dimgrey")

    ax2.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax2.axhline(+model.rmse_loo, color="grey", ls=":", lw=0.8)
    ax2.axhline(-model.rmse_loo, color="grey", ls=":", lw=0.8)
    for q, col in q_colors.items():
        ax2.scatter([], [], color=col, label=f"q_acid={q:+d}", s=50)
    ax2.legend(fontsize=8)
    ax2.set_xlabel("pKa_calc  (ΔG_DFT / RT ln10)", fontsize=11)
    ax2.set_ylabel("pKa_pred − pKa_exp", fontsize=11)
    ax2.set_title("Residuals vs pKa_calc\n(dotted lines = ±RMSE)", fontsize=10)
    ax2.grid(alpha=0.25)

    fig.suptitle(
        f"pKa GPR Calibration — {method} / CPCM(Water) + GFN2-xTB RRHO\n"
        f"N={model.n_train} ({model.n_full} full descriptors, {model.n_partial} partial)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    logging.info("GPR plot saved → %s", out_path)
    plt.close(fig)


def check_gpr_model(gpr_json: Path) -> None:
    """Print summary of an existing GPR model."""
    if not gpr_json.exists():
        print(f"ERROR: {gpr_json} does not exist.")
        return

    data  = json.loads(gpr_json.read_text())
    model = GPRModel.from_dict(data["model"])

    print(f"GPR model:  {gpr_json}")
    print(f"Method:     {data.get('method', 'unknown')}")
    print(f"Features:   {data['model'].get('feature_names', 'unknown')}")
    print(f"Mode:       {model.feature_mode}")
    print(f"N_train:    {model.n_train}  ({model.n_full} full, {model.n_partial} partial)")
    print(f"LOO RMSE:   {model.rmse_loo:.4f} pKa units")
    print(f"LOO MAE:    {model.mae_loo:.4f} pKa units")
    print(f"Noise σ:    {model.alpha_noise:.4f} pKa units")
    print(f"Amplitude:  {model.amplitude:.4f}")
    print()
    feats = data["model"].get("feature_names", model.FEATURE_NAMES)
    ls    = model.length_scales
    print(f"  {'Feature':>15}  {'Length scale':>13}")
    print("  " + "─" * 32)
    for f, l in zip(feats, ls):
        print(f"  {f:>15}  {l:13.4f}")
    print()
    if model.n_partial > 0:
        print(f"  NOTE: {model.n_partial} training step(s) used partial descriptors.")
        print("  Run pka_calibrate.py with --outdir matching these calculations")
        print("  to extract full descriptors and improve the model.")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _write_predictions_report(
        out_dir: Path,
        gpr_path: Path,
        method_kw: str,
        completed: list[dict],
        model,
) -> None:
    """
    Write (or overwrite) gpr_predictions_test.txt with all completed steps.
    Each entry in `completed` is a dict from _predict_one_compound().
    """
    hdr = (f"{'Compound':22s}  {'Step':24s}  {'q':>4}  {'pKa_calc':>9}  "
           f"{'pKa_exp':>8}  {'pKa_pred':>9}  {'σ':>6}  {'err':>6}")
    sep = "─" * 100

    all_errors = []
    lines = [
        f"GPR Test Set Predictions",
        f"Model: {gpr_path}",
        f"Method: {method_kw}",
        "",
        f"{'Compound':22s}  {'Step':24s}  {'q':>6}  {'pKa_calc':>9}  "
        f"{'pKa_exp':>8}  {'pKa_pred':>9}  {'σ':>6}  {'err':>6}",
        sep,
    ]
    for cdata in completed:
        for row in cdata["rows"]:
            lines.append(row["txt"])
            if row["err"] is not None:
                all_errors.append(row["err"])
    lines.append(sep)
    if all_errors:
        arr  = np.array(all_errors)
        rmse = float(np.sqrt(np.mean(arr ** 2)))
        mae  = float(np.mean(np.abs(arr)))
        n    = len(arr)
        lines.append(f"\nTest RMSE: {rmse:.3f}   MAE: {mae:.3f}   "
                     f"N={n} steps with experimental pKa")
        lines.append(f"Training LOO RMSE: {model.rmse_loo:.3f}   "
                     f"LOO MAE: {model.mae_loo:.3f}")
    report_path = out_dir / "gpr_predictions_test.txt"
    report_path.write_text("\n".join(lines) + "\n")


def _predict_one_compound(
        msf_path: Path,
        out_dir: Path,
        model,
        method_kw: str,
        method_kw2: str,
        args,
) -> dict:
    """
    Run ORCA (if needed), build descriptors, predict pKa for one compound.
    Returns a dict suitable for accumulation and progressive reporting.
    """
    compound = msf_path.stem
    logging.info("")
    logging.info("── %s ──", compound)

    g_cache = load_cached_G_eff(msf_path, out_dir)
    if g_cache is None:
        run_orca_for_compound(
            msf_path=msf_path, out_dir=out_dir,
            orca_binary=args.orca, nprocs=args.nprocs,
            nconf=args.nconf, ewindow=args.ewindow,
            method=method_kw2, do_tautomers=not args.no_tautomers,
            temperature=args.temp, dry_run=args.dry_run,
        )

    steps = build_step_descriptors(msf_path, out_dir, args.temp,
                                   dry_run=args.dry_run,
                                   orca_binary=args.orca, nprocs=args.nprocs)
    n_full = sum(1 for s in steps if s.feature_vector(model.charge_means) is not None)
    logging.info("  → %d step(s), %d with full descriptors.", len(steps), n_full)

    sep = "─" * 100
    rows = []
    for s in steps:
        pred, sigma = model.predict(s)
        err = (s.pka_exp - pred) if (s.pka_exp is not None and not np.isnan(s.pka_exp)) \
              else None
        pka_exp_str = f"{s.pka_exp:.3f}" if s.pka_exp is not None else "  n/a "
        err_str     = f"{err:+.3f}" if err is not None else "  n/a"
        step_label  = f"{s.acid_name[:12]}→{s.base_name[:10]}"
        txt = (f"{s.compound:22s}  {step_label:24s}  {s.acid_charge:>4d}  "
               f"{s.pka_calc:>9.2f}  {pka_exp_str:>8s}  {pred:>9.3f}  "
               f"{sigma:>6.3f}  {err_str:>6}")
        rows.append({
            "step": s, "pred": pred, "sigma": sigma, "err": err, "txt": txt,
        })

    # Build microstate data for population plot
    mss = parse_msf(msf_path)
    g   = load_cached_G_eff(msf_path, out_dir)
    if g:
        for ms in mss:
            ms.G_eff = g.get(ms.name)

    cdata = {
        "name":        compound,
        "microstates": mss,
        "steps":       steps,
        "rows":        rows,
        "pKa_preds":   [(r["pred"],  f"{r['step'].acid_name}→{r['step'].base_name}")
                        for r in rows],
        "pKa_sigmas":  [(r["sigma"], f"{r['step'].acid_name}→{r['step'].base_name}")
                        for r in rows],
        "pKa_exps":    [(r["step"].pka_exp,
                         f"{r['step'].acid_name}→{r['step'].base_name}")
                        for r in rows if r["step"].pka_exp is not None],
    }
    return cdata


def _run_predict_mode(args) -> None:
    """
    Apply an existing GPR model to new (test) compounds.

    Processes compounds one at a time. After each compound completes:
      - Appends results to gpr_predictions_test.txt (full overwrite, all
        completed so far)
      - Regenerates gpr_predictions_test.png with all completed compounds
      - Writes a per-compound population plot:
        gpr_pop_<compound>.png

    This way long-running ORCA jobs don't leave you blind — results for
    finished compounds are available immediately.
    """
    gpr_path = Path(args.gpr)
    out_dir  = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, method_kw = load_gpr_model(gpr_path)
    logging.info("=" * 62)
    logging.info("pka_gpr_calibrate.py  —  PREDICT MODE")
    logging.info("  GPR model:  %s", gpr_path)
    logging.info("  Method:     %s", method_kw)
    logging.info("  Compounds:  %d .msf files", len(args.msf_files))
    logging.info("  Work dir:   %s", out_dir)
    logging.info("=" * 62)

    method_kw2 = {
        "r2scan3c": "r2SCAN-3c", "r2scan-3c": "r2SCAN-3c",
        "pbe3c":    "PBE-3c",    "pbe-3c":    "PBE-3c",
    }.get(args.method.lower(), args.method)

    completed: list[dict] = []   # accumulates per-compound results

    for msf_file in args.msf_files:
        msf_path = Path(msf_file)
        if not msf_path.exists():
            logging.error("Not found: %s — skipping.", msf_path)
            continue

        try:
            cdata = _predict_one_compound(
                msf_path, out_dir, model, method_kw, method_kw2, args)
        except Exception as exc:
            logging.error("[%s] Failed: %s", msf_path.stem, exc)
            continue

        completed.append(cdata)

        # ── 1. Update text report ─────────────────────────────────────────────
        _write_predictions_report(out_dir, gpr_path, method_kw, completed, model)
        print(f"  ✓ {msf_path.stem}: "
              f"{len(cdata['rows'])} step(s)  "
              + "  ".join(f"pKa_pred={r['pred']:.2f}(±{r['sigma']:.2f})"
                           for r in cdata["rows"]))
        print(f"  → Report updated: {out_dir / 'gpr_predictions_test.txt'}")

        # ── 2. Regenerate overall scatter plot ────────────────────────────────
        all_steps = [r["step"]  for c in completed for r in c["rows"]]
        all_preds = np.array([r["pred"]  for c in completed for r in c["rows"]])
        all_sigs  = np.array([r["sigma"] for c in completed for r in c["rows"]])
        try:
            plot_path = out_dir / "gpr_predictions_test.png"
            _plot_test_predictions(model, all_steps, all_preds, all_sigs,
                                   plot_path.resolve(), method_kw)
            print(f"  → Scatter plot updated: {plot_path}")
        except Exception as exc:
            logging.warning("Scatter plot failed: %s", exc)

        # ── 3. Per-compound population plot ───────────────────────────────────
        pop_path = out_dir / f"gpr_pop_{msf_path.stem}.png"
        try:
            _plot_microstate_populations(
                [cdata], pop_path.resolve(), method_kw, T=args.temp)
            print(f"  → Population plot: {pop_path}")
        except Exception as exc:
            logging.warning("Population plot for %s failed: %s",
                             msf_path.stem, exc)

    # ── Final summary ─────────────────────────────────────────────────────────
    if not completed:
        logging.error("No compounds processed successfully.")
        sys.exit(1)

    all_errors = [r["err"] for c in completed for r in c["rows"]
                  if r["err"] is not None]
    sep = "─" * 100
    print()
    print(sep)
    print("  GPR Predictions on Test Set")
    print(f"  Model: {gpr_path}  ({method_kw})")
    print(sep)
    hdr = (f"{'Compound':22s}  {'Step':24s}  {'q':>6}  {'pKa_calc':>9}  "
           f"{'pKa_exp':>8}  {'pKa_pred':>9}  {'σ':>6}  {'err':>6}")
    print("  " + hdr)
    print("  " + sep)
    for c in completed:
        for r in c["rows"]:
            print("  " + r["txt"])
    print("  " + sep)

    if all_errors:
        arr  = np.array(all_errors)
        rmse = float(np.sqrt(np.mean(arr ** 2)))
        mae  = float(np.mean(np.abs(arr)))
        print(f"\n  Test set RMSE: {rmse:.3f}  MAE: {mae:.3f}  "
              f"(N={len(arr)} steps with experimental pKa)")
        print(f"  Training LOO RMSE: {model.rmse_loo:.3f}  LOO MAE: {model.mae_loo:.3f}")
        if rmse > model.rmse_loo * 1.5:
            print("  ⚠  Test RMSE > 1.5 × LOO RMSE — may be outside training distribution.")
        else:
            print("  ✓  Test RMSE within expected range of LOO RMSE.")

    print(f"\n  Report: {out_dir / 'gpr_predictions_test.txt'}")
    print(f"  Plot:   {out_dir / 'gpr_predictions_test.png'}")
    print(f"  Population plots: {out_dir}/gpr_pop_<compound>.png")



def _compute_microstate_populations(
        microstates,
        T: float = 298.15,
        pH_range: np.ndarray | None = None,
        pKa_pred: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute macrostate Boltzmann populations across a pH range.

    Uses calibrated (GPR-predicted) pKa values rather than raw G_eff.

    Raw G_eff values correspond to pKa_calc ≈ 190–230, meaning the DFT
    free energy differences between microstates are ~0.4–0.5 Eh ≈ 200 pKa
    units. The Boltzmann exponent ΔG/kT ≈ 450 causes complete numerical
    underflow (exp(-450) = 0), so using G_eff directly always shows one
    state at 100% for all pH 0–14.

    The correct approach: compute α_i analytically from the macro-pKa
    transitions using the standard polyprotic species distribution formula:

        log10(α_i) = Σ_{j<i} (pKa_j − pH)   (microstates ordered most→least protonated)
        α_i = 10^(log10_α_i − max) / Σ_k 10^(log10_α_k − max)

    This gives smooth sigmoidal transitions at exactly the predicted pKa values,
    identical to the AcepKa / Uni-pKa microstate population diagrams.

    Args:
        microstates: list[Microstate] ordered by n_protons descending
        pKa_pred:    list of predicted pKa values for each step (len = n_ms − 1)
        pH_range:    pH values to evaluate at
    Returns:
        pH_arr  : shape (n_pH,)
        pop_arr : shape (n_ms, n_pH), fractions summing to 1 at each pH
    """
    if pH_range is None:
        pH_range = np.linspace(0, 14, 281)
    if pKa_pred is None or len(pKa_pred) == 0:
        # Fallback: uniform distribution
        n = len(microstates)
        return pH_range, np.ones((n, len(pH_range))) / n

    # Sort microstates by n_protons descending (most protonated = index 0)
    sorted_ms = sorted(microstates, key=lambda m: m.n_protons, reverse=True)
    pKa_arr   = np.asarray(pKa_pred, dtype=float)
    n_ms      = len(sorted_ms)
    n_pH      = len(pH_range)

    # log10(α_i) = Σ_{j=i}^{n_ms-2} (pKa_j − pH)
    #
    # Derivation (polyprotic species distribution):
    #   α_0 (most protonated) dominates at low pH.
    #   α_{n-1} (least protonated) dominates at high pH.
    #   At pH = pKa_i: α_i = α_{i+1} = 50% (all others ≈ 0%).
    #
    # Cumulative sum from i to end (not 0 to i-1):
    #   α_0: all pKa terms  → most positive log10 at low pH → dominates
    #   α_1: pKa_1..pKa_{n-2} terms
    #   ...
    #   α_{n-1}: 0 terms (reference)
    log10_alpha = np.zeros((n_ms, n_pH))
    for i in range(n_ms):
        for j in range(i, n_ms - 1):
            log10_alpha[i] += (pKa_arr[j] - pH_range)

    # Normalise in log10 space (immune to overflow/underflow)
    log10_max = log10_alpha.max(axis=0, keepdims=True)
    alpha = 10.0 ** (log10_alpha - log10_max)
    alpha /= alpha.sum(axis=0, keepdims=True)

    return pH_range, alpha


def _plot_microstate_populations(
        compound_data: list[dict],
        out_path: Path,
        method: str,
        T: float = 298.15,
) -> None:
    """
    One subplot per compound showing microstate population (%) vs pH.

    For each microstate curve the GPR prediction uncertainty (σ) is shown
    as a shaded band: populations are computed at pKa_pred − σ and
    pKa_pred + σ and the area between them is filled.  This visualises how
    a ±1σ shift of every pKa transition propagates into population uncertainty.

    Vertical lines:
      black dashed  — GPR-predicted pKa (nominal)
      black dotted  — GPR ±1σ envelope (pKa ± σ)
      red dotted    — experimental pKa (when available)
      grey dash-dot — physiological pH 7.4
    """
    import matplotlib.pyplot as plt

    n     = len(compound_data)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.2 * nrows),
                             squeeze=False)

    ms_colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    pH_arr     = np.linspace(0, 14, 281)

    for idx, cdata in enumerate(compound_data):
        ax   = axes[idx // ncols][idx % ncols]
        name = cdata["name"]
        mss  = cdata["microstates"]

        sorted_mss = sorted(mss, key=lambda m: m.n_protons, reverse=True)
        pKa_preds  = [pka   for pka,   _ in cdata.get("pKa_preds",  [])]
        pKa_sigmas = [sigma for sigma, _ in cdata.get("pKa_sigmas", [])]
        pKa_exps   = cdata.get("pKa_exps", [])

        if not pKa_preds:
            ax.text(0.5, 0.5, "No pKa predictions", ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='grey')
            ax.set_title(name, fontsize=10)
            continue

        # ── Three population curves: nominal, low (−σ), high (+σ) ─────────────
        _, pop_nom  = _compute_microstate_populations(
            sorted_mss, T=T, pH_range=pH_arr, pKa_pred=pKa_preds)

        # pKa_sigmas may be shorter than pKa_preds if σ was unavailable
        if pKa_sigmas and len(pKa_sigmas) == len(pKa_preds):
            pKa_low  = [p - s for p, s in zip(pKa_preds, pKa_sigmas)]
            pKa_high = [p + s for p, s in zip(pKa_preds, pKa_sigmas)]
            _, pop_low  = _compute_microstate_populations(
                sorted_mss, T=T, pH_range=pH_arr, pKa_pred=pKa_low)
            _, pop_high = _compute_microstate_populations(
                sorted_mss, T=T, pH_range=pH_arr, pKa_pred=pKa_high)
        else:
            pop_low  = pop_nom
            pop_high = pop_nom

        for i, (ms, col) in enumerate(zip(sorted_mss, ms_colours)):
            label = f"{ms.name}  (q={ms.charge:+d})"
            # Uncertainty band (envelope from pKa±σ)
            lo  = np.minimum(pop_low[i],  pop_high[i]) * 100.0
            hi  = np.maximum(pop_low[i],  pop_high[i]) * 100.0
            nom = pop_nom[i] * 100.0
            ax.fill_between(pH_arr, lo, hi, color=col, alpha=0.15, linewidth=0)
            ax.plot(pH_arr, nom, color=col, lw=2.0, label=label)

        # ── Vertical markers ──────────────────────────────────────────────────
        # GPR ±σ envelope (light dotted) — one pair per step
        for (pka, _), (sigma, _) in zip(cdata.get("pKa_preds", []),
                                         cdata.get("pKa_sigmas", [])):
            for shift in (-sigma, +sigma):
                ax.axvline(pka + shift, color='#555555', lw=0.7,
                           ls=':', alpha=0.55)

        # GPR nominal pKa (dashed black)
        for pka_pred, _ in cdata.get("pKa_preds", []):
            ax.axvline(pka_pred, color='black', lw=1.2, ls='--', alpha=0.75)
            ax.text(pka_pred + 0.12, 93,
                    f'pred\n{pka_pred:.2f}',
                    fontsize=6, color='black', va='top', alpha=0.85)

        # Experimental pKa (dotted red)
        for pka_exp, _ in pKa_exps:
            ax.axvline(pka_exp, color='#cc0000', lw=1.0, ls=':', alpha=0.85)
            ax.text(pka_exp + 0.12, 73,
                    f'exp\n{pka_exp:.2f}',
                    fontsize=6, color='#cc0000', va='top', alpha=0.85)

        # Physiological pH reference
        ax.axvline(7.4, color='grey', lw=0.8, ls='-.', alpha=0.45)
        ax.text(7.52, 48, 'pH 7.4', fontsize=5.5, color='grey',
                alpha=0.65, rotation=90)

        ax.set_xlim(0, 14)
        ax.set_ylim(-2, 105)
        ax.set_xlabel('pH', fontsize=10)
        ax.set_ylabel('Population (%)', fontsize=10)
        ax.set_title(name.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, framealpha=0.85, loc='upper right',
                  handlelength=1.5, borderpad=0.5)
        ax.grid(alpha=0.18)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f'Microstate Populations vs pH  (shaded = ±1σ GPR uncertainty)\n'
        f'{method} / CPCM(Water) + GFN2-xTB RRHO',
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    if not out_path.exists() or out_path.stat().st_size < 1024:
        raise RuntimeError(
            f"savefig appeared to succeed but {out_path} is missing or empty "
            f"({out_path.stat().st_size if out_path.exists() else 0} bytes). "
            f"Check disk space and write permissions.")
    logging.info("Microstate population plot saved → %s  (%d KB)",
                 out_path, out_path.stat().st_size // 1024)


def _plot_test_predictions(
        model: GPRModel,
        steps: list[StepDescriptors],
        preds: np.ndarray,
        sigmas: np.ndarray,
        out_path: Path,
        method: str,
) -> None:
    """Plot predicted vs experimental pKa for the test set."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    charge_colors = {-2: '#d62728', -1: '#ff7f0e', 0: '#2ca02c',
                     1: '#1f77b4', 2: '#9467bd', 3: '#8c564b'}

    steps_with_exp = [(s, p, sig) for s, p, sig in zip(steps, preds, sigmas)
                      if s.pka_exp is not None]

    if steps_with_exp:
        pka_exp_vals  = np.array([s.pka_exp for s, _, _ in steps_with_exp])
        pka_pred_vals = np.array([p for _, p, _ in steps_with_exp])
        sigma_vals    = np.array([sig for _, _, sig in steps_with_exp])
        residuals     = pka_pred_vals - pka_exp_vals

        # Left: predicted vs exp
        for s, pred, sigma in steps_with_exp:
            col = charge_colors.get(s.acid_charge, 'grey')
            ax1.errorbar(s.pka_exp, pred, yerr=sigma, fmt='o',
                         color=col, alpha=0.8, capsize=3, markersize=7)
            ax1.annotate(s.compound, (s.pka_exp, pred),
                         fontsize=6, alpha=0.7,
                         xytext=(4, 2), textcoords='offset points')

        lo = min(pka_exp_vals.min(), pka_pred_vals.min()) - 0.5
        hi = max(pka_exp_vals.max(), pka_pred_vals.max()) + 0.5
        ax1.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='ideal')
        ax1.set_xlabel('pKa_exp', fontsize=12)
        ax1.set_ylabel('pKa_pred ± σ', fontsize=12)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae  = float(np.mean(np.abs(residuals)))
        ax1.set_title(f'GPR Test Predictions\nRMSE={rmse:.3f}  MAE={mae:.3f}', fontsize=11)
        ax1.grid(alpha=0.25)

        # Legend for charge states
        seen = set()
        for s, _, _ in steps_with_exp:
            q = s.acid_charge
            if q not in seen:
                col = charge_colors.get(q, 'grey')
                ax1.plot([], [], 'o', color=col, label=f'q_acid={q:+d}')
                seen.add(q)
        ax1.legend(fontsize=9, framealpha=0.9)

        # Right: residuals
        ax2.axhline(0, color='k', lw=1, ls='--')
        ax2.axhline( rmse, color='grey', lw=1, ls=':', alpha=0.7)
        ax2.axhline(-rmse, color='grey', lw=1, ls=':', alpha=0.7)
        for s, pred, _ in steps_with_exp:
            col = charge_colors.get(s.acid_charge, 'grey')
            resid = pred - s.pka_exp
            ax2.scatter(s.pka_calc, resid, color=col, s=60, alpha=0.8, zorder=3)
            ax2.annotate(s.compound, (s.pka_calc, resid),
                         fontsize=6, alpha=0.7,
                         xytext=(3, 2), textcoords='offset points')
        ax2.set_xlabel('pKa_calc  (ΔG_DFT / RT ln10)', fontsize=12)
        ax2.set_ylabel('pKa_pred − pKa_exp', fontsize=12)
        ax2.set_title(f'Residuals vs pKa_calc\n(dotted = ±RMSE)', fontsize=11)
        ax2.grid(alpha=0.25)

    fig.suptitle(
        f'GPR Test Set — {method} / CPCM(Water) + GFN2-xTB RRHO\n'
        f'N={len(steps_with_exp)} steps  '
        f'(Training LOO RMSE={model.rmse_loo:.3f})',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        prog="pka_gpr_calibrate.py",
        description="pKa prediction via DFT composite method + GPR calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Modes
            -----
            --train    Run ORCA (if needed) + build descriptors + train GPR.
                       Produces gpr_model.json, gpr_report.txt, gpr_calibration_plot.png.

            --predict  Run ORCA (if needed) + build descriptors + apply saved GPR.
                       Requires --gpr. Produces prediction table, scatter plot,
                       and per-compound microstate population plot with σ bands.

            --check    Print a summary of a saved gpr_model.json and exit.

            Examples
            --------
            # Train on labelled compounds:
            python pka_gpr_calibrate.py training_set/*.msf --train \\
                --outdir pka_results \\
                --orca /apps/software/orca/orca-6.1.0/orca --nprocs 56

            # Predict on new compounds:
            python pka_gpr_calibrate.py test_set/*.msf --predict \\
                --gpr pka_results/gpr_model.json \\
                --outdir pka_results_test \\
                --orca /apps/software/orca/orca-6.1.0/orca --nprocs 56

            # Inspect a saved model:
            python pka_gpr_calibrate.py --check pka_results/gpr_model.json
        """),
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train",   action="store_true",
                      help="Train mode: run ORCA if needed, build descriptors, fit GPR.")
    mode.add_argument("--predict", action="store_true",
                      help="Predict mode: run ORCA if needed, apply saved GPR model. "
                           "Requires --gpr.")
    mode.add_argument("--check",   metavar="GPR_JSON", nargs="?", const="__check__",
                      help="Print summary of a saved gpr_model.json and exit.")

    p.add_argument("msf_files", nargs="*", metavar="COMPOUND.msf")
    p.add_argument("--gpr",     metavar="GPR_JSON", default=None,
                   help="Path to gpr_model.json (required for --test).")
    p.add_argument("--orca",    default="orca",
                   help="Path to ORCA binary (default: 'orca' on PATH).")
    p.add_argument("--nprocs",  type=int,   default=4,
                   help="MPI processes for ORCA (default: 4).")
    p.add_argument("--nconf",   type=int,   default=10,
                   help="Max GOAT conformers per microstate (default: 10).")
    p.add_argument("--ewindow", type=float, default=3.0,
                   help="Energy window for conformer selection, kcal/mol (default: 3.0).")
    p.add_argument("--method",  default="r2scan3c",
                   help="DFT method keyword (default: r2scan3c).")
    p.add_argument("--temp",    type=float, default=TEMPERATURE,
                   help=f"Temperature in K (default: {TEMPERATURE}).")
    p.add_argument("--outdir",  default="pka_results",
                   help="Output directory (default: pka_results).")
    p.add_argument("--no-tautomers", action="store_true",
                   help="Disable tautomer search in GOAT.")
    p.add_argument("--dry-run", action="store_true",
                   help="Mock ORCA outputs for testing (no calculations run).")

    args = p.parse_args()

    # ── --check mode ─────────────────────────────────────────────────────────
    if args.check:
        gpr_json = args.check if args.check != "__check__" else (args.gpr or "")
        if not gpr_json:
            # --check with no value and no --gpr: try default location
            gpr_json = Path(args.outdir) / "gpr_model.json"
        check_gpr_model(Path(gpr_json))
        return

    if not args.msf_files:
        p.error("Provide one or more COMPOUND.msf files.")

    # ── --predict mode ────────────────────────────────────────────────────────
    if args.predict:
        if not args.gpr:
            p.error("--predict requires --gpr <path/to/gpr_model.json>")
        _run_predict_mode(args)
        return

    # ── --train mode ──────────────────────────────────────────────────────────
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_kw = {
        "r2scan3c": "r2SCAN-3c", "r2scan-3c": "r2SCAN-3c",
        "pbe3c":    "PBE-3c",    "pbe-3c":    "PBE-3c",
    }.get(args.method.lower(), args.method)

    logging.info("=" * 62)
    logging.info("pka_gpr_calibrate.py  —  TRAIN MODE")
    logging.info("  Method:    %s / CPCM(Water) + GFN2-xTB RRHO", method_kw)
    logging.info("  Compounds: %d .msf files", len(args.msf_files))
    logging.info("  Output:    %s", out_dir)
    logging.info("=" * 62)

    all_steps: list[StepDescriptors] = []

    for msf_file in args.msf_files:
        msf_path = Path(msf_file)
        if not msf_path.exists():
            logging.error("Not found: %s — skipping.", msf_path)
            continue

        logging.info("")
        logging.info("── %s ──", msf_path.stem)

        # Run ORCA if G_eff not already cached
        g_cache = load_cached_G_eff(msf_path, out_dir)
        if g_cache is None:
            run_orca_for_compound(
                msf_path=msf_path, out_dir=out_dir,
                orca_binary=args.orca, nprocs=args.nprocs,
                nconf=args.nconf, ewindow=args.ewindow,
                method=method_kw, do_tautomers=not args.no_tautomers,
                temperature=args.temp, dry_run=args.dry_run,
            )

        steps = build_step_descriptors(
            msf_path, out_dir, args.temp, dry_run=args.dry_run,
            orca_binary=args.orca, nprocs=args.nprocs)
        all_steps.extend(steps)
        n_full = sum(1 for s in steps if s.feature_vector() is not None)
        logging.info("  → %d step(s), %d with full descriptors.", len(steps), n_full)

    if not all_steps:
        logging.error("No training steps collected. Check that .msf files have "
                      "pka_step annotations.")
        sys.exit(1)

    n_full_total = sum(1 for s in all_steps if s.feature_vector() is not None)
    logging.info("")
    logging.info("Total: %d training steps, %d with full descriptors.",
                 len(all_steps), n_full_total)

    # ── Train GPR ─────────────────────────────────────────────────────────────
    logging.info("Training GPR model…")
    model = GPRModel()
    model.fit(all_steps)

    # ── Save outputs ──────────────────────────────────────────────────────────
    gpr_path    = out_dir / "gpr_model.json"
    report_path = out_dir / "gpr_report.txt"
    plot_path   = out_dir / "gpr_calibration_plot.png"

    save_gpr_model(model, all_steps, method_kw, gpr_path)
    write_gpr_report(model, all_steps, method_kw, report_path)
    plot_gpr(model, all_steps, plot_path, method_kw)

    print()
    print("=" * 62)
    print("  GPR Training Complete")
    print(f"  {method_kw} / CPCM(Water) + GFN2-xTB RRHO")
    print("=" * 62)
    print(f"  Training steps:   {model.n_train}")
    print(f"  Full descriptors: {model.n_full}")
    print(f"  Partial only:     {model.n_partial}")
    print(f"  LOO RMSE:  {model.rmse_loo:.4f} pKa units")
    print(f"  LOO MAE:   {model.mae_loo:.4f} pKa units")
    print()
    print(f"  Model:  {gpr_path}")
    print(f"  Report: {report_path}")
    print(f"  Plot:   {plot_path}")
    print()
    print("  Next: predict on new compounds with --predict")
    print(f"    python pka_gpr_calibrate.py test_set/*.msf --predict \\")
    print(f"        --gpr {gpr_path} --outdir pka_results_test --orca ...")


if __name__ == "__main__":
    main()