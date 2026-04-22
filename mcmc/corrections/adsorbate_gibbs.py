"""Three-layer free-energy correction for surface Pourbaix slabs.

Scheme:
    G_corr = E_DFT + A + B + C

    A) Bulk oxide correction   : Δ_O × (number of O atoms in slab)
                                 Fit to experimental ΔG°_f of the bulk oxide
                                 reference (e.g. IrO₂). Applied to ALL O atoms.

    B) Adsorbate Gibbs         : Σᵢ nᵢ · (ZPE + ∫CpdT - TS)ᵢ
                                 Adsorbates identified by Ir-coordination:
                                   coord_Ir == 1  → adsorbate
                                   coord_Ir >= 2  → bulk O, not an adsorbate
                                 Sub-species (*O / *OH / *OOH / *H₂O) from
                                 O-H and O-O connectivity.

    C) Hydrogen bonds          : ε_HB × (number of H-bonds)
                                 H-bonds identified by MDAnalysis
                                 (Luzar-Chandler / Baker-Hubbard criterion).

Intrinsic values are literature defaults; override via the keyword arguments of
``slab_correction`` for system-specific phonon calculations.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import ase
import numpy as np


# ---------------------------------------------------------------------------
# Defaults (override per-system if you have your own phonon data)
# ---------------------------------------------------------------------------

#: Per-adsorbate intrinsic Gibbs correction (ZPE + ∫Cp dT − TS) at 298 K, 1 atm.
#: Excludes H-bonding, which is counted separately. Sources: Peterson 2010 EES,
#: Rong & Kolpak 2015 JPCL, Calle-Vallejo 2011.
ADS_G_INTRINSIC: dict[str, float] = {
    "O":   0.065 + 0.020 - 0.000,   # = +0.085 eV
    "OH":  0.360 + 0.050 - 0.070,   # = +0.340 eV
    "OOH": 0.430 + 0.080 - 0.170,   # = +0.340 eV
    "H2O": 0.570 + 0.100 - 0.670,   # = +0.000 eV
    "H":   0.140 + 0.010 - 0.000,   # = +0.150 eV
}

#: Per-O oxide correction fit against experimental ΔG°_f(IrO₂, s) = −2.017 eV/fu
#: using the user's own DFT (ENCUT=520, VASP PAW-PBE, AqCompat on gas refs).
DEFAULT_DELTA_O: float = -0.1252  # eV per O atom

#: Per-H-bond correction. Empirical DFT-error correction for PBE H-bonds.
DEFAULT_EPS_HBOND: float = -0.15  # eV per counted H-bond


# ---------------------------------------------------------------------------
# Layer B: adsorbate identification by Ir-coordination
# ---------------------------------------------------------------------------

def count_adsorbates(
    atoms: ase.Atoms,
    metal_symbol: str = "Ir",
    r_MO: float = 2.3,
    r_OH: float = 1.2,
    r_OO: float = 1.6,
) -> Counter:
    """Identify and count surface adsorbates from geometry.

    An O atom is classified as an adsorbate if its coordination number to
    the metal (Ir) is exactly 1. O atoms with two or more metal neighbors
    are treated as bulk lattice O. A terminal peroxo O (coord_M == 0) is
    paired with its inner O partner to form *OOH.

    Args:
        atoms: ASE Atoms (relaxed geometry, with PBC).
        metal_symbol: Surface metal symbol (default Ir).
        r_MO: Metal-oxygen bond cutoff (Å).
        r_OH: Oxygen-hydrogen covalent cutoff (Å).
        r_OO: Oxygen-oxygen peroxo cutoff (Å).

    Returns:
        Counter mapping species ('O', 'OH', 'OOH', 'H2O') to count.
    """
    sym = atoms.get_chemical_symbols()
    O_idx = [i for i, s in enumerate(sym) if s == "O"]
    H_idx = [i for i, s in enumerate(sym) if s == "H"]
    M_idx = [i for i, s in enumerate(sym) if s == metal_symbol]

    def neighbors_within(i: int, candidates: Iterable[int], r_cut: float) -> list[int]:
        return [
            j for j in candidates
            if j != i and atoms.get_distance(i, j, mic=True) < r_cut
        ]

    O_to_M = {i: neighbors_within(i, M_idx, r_MO) for i in O_idx}
    O_to_H = {i: neighbors_within(i, H_idx, r_OH) for i in O_idx}
    O_to_O = {i: neighbors_within(i, O_idx, r_OO) for i in O_idx}

    counts: Counter = Counter()
    used: set[int] = set()

    # First pass: *OOH (inner with 1 metal + 1 terminal O that has 0 metal + 1 H)
    for i in O_idx:
        if i in used:
            continue
        if len(O_to_M[i]) == 1 and len(O_to_O[i]) == 1:
            j = O_to_O[i][0]
            if len(O_to_M[j]) == 0 and len(O_to_H[j]) == 1:
                counts["OOH"] += 1
                used.update((i, j))

    # Second pass: *O / *OH / *H2O (singly-coordinated O)
    for i in O_idx:
        if i in used:
            continue
        n_M = len(O_to_M[i])
        if n_M >= 2:
            continue  # bulk lattice O
        if n_M == 1:
            n_H = len(O_to_H[i])
            n_O = len(O_to_O[i])
            if n_O > 0:
                continue  # partner already consumed or unknown pairing
            if n_H == 0:
                counts["O"] += 1
            elif n_H == 1:
                counts["OH"] += 1
            elif n_H >= 2:
                counts["H2O"] += 1
            used.add(i)

    return counts


# ---------------------------------------------------------------------------
# Layer C: H-bond counting via MDAnalysis (standard Luzar-Chandler criterion)
# ---------------------------------------------------------------------------

def _ase_to_mda(atoms: ase.Atoms):
    """Minimal ASE → MDAnalysis Universe conversion for a single frame."""
    import MDAnalysis as mda

    n = len(atoms)
    u = mda.Universe.empty(n, trajectory=True)
    symbols = list(atoms.get_chemical_symbols())
    u.add_TopologyAttr("name", symbols)
    u.add_TopologyAttr("type", symbols)
    u.add_TopologyAttr("resid", [1])
    u.add_TopologyAttr("resname", ["SYS"])
    u.add_TopologyAttr("segid", ["A"])
    u.atoms.positions = atoms.get_positions()
    u.dimensions = atoms.get_cell().cellpar()
    return u


def count_hbonds_ase(
    atoms: ase.Atoms,
    d_a_cutoff: float = 3.0,
    d_h_a_angle_cutoff: float = 120.0,
    d_h_cutoff: float = 1.2,
) -> int:
    """Pure-ASE/numpy fallback implementing the same Luzar-Chandler criterion
    as ``count_hbonds``, without MDAnalysis. Used when MDAnalysis isn't
    available in the env (e.g. Expanse my_pymatgen).

    Algorithm:
      - For each H atom, find its donor O (nearest O within d_h_cutoff).
        If no donor O → not a valid H-bond donor.
      - For each donor-H, find acceptor Os within d_a_cutoff (excluding donor).
      - For each candidate acceptor, compute O-H...O angle; if ≥ angle cutoff,
        count as one H-bond.
    """
    syms = atoms.get_chemical_symbols()
    O_idx = [i for i, s in enumerate(syms) if s == "O"]
    H_idx = [i for i, s in enumerate(syms) if s == "H"]
    if not H_idx or len(O_idx) < 2:
        return 0

    n_hb = 0
    for h in H_idx:
        # Donor O: nearest O within covalent cutoff
        d_oh = atoms.get_distances(h, O_idx, mic=True)
        close = [(d_oh[k], O_idx[k]) for k in range(len(O_idx)) if d_oh[k] < d_h_cutoff]
        if not close:
            continue
        _, donor = min(close)
        # Candidate acceptors: O atoms within d_a_cutoff of donor, excluding donor
        others = [o for o in O_idx if o != donor]
        if not others:
            continue
        d_don_other = atoms.get_distances(donor, others, mic=True)
        for k, acceptor in enumerate(others):
            if d_don_other[k] > d_a_cutoff:
                continue
            # Angle donor-H...acceptor (mic-aware via get_angle)
            try:
                angle = atoms.get_angle(donor, h, acceptor, mic=True)
            except Exception:
                continue
            if angle >= d_h_a_angle_cutoff:
                n_hb += 1
    return n_hb


def count_hbonds(
    atoms: ase.Atoms,
    d_a_cutoff: float = 3.0,
    d_h_a_angle_cutoff: float = 120.0,
    d_h_cutoff: float = 1.2,
) -> int:
    """Count hydrogen bonds using MDAnalysis HydrogenBondAnalysis if available,
    else fall back to ``count_hbonds_ase`` (pure ASE/numpy equivalent).

    Implements the standard Luzar-Chandler / Baker-Hubbard criterion:
    donor O covalently bonded to H (``d_O-H < d_h_cutoff``), acceptor O
    within ``d_a_cutoff`` of the donor, and O-H...O angle greater than
    ``d_h_a_angle_cutoff``.

    Args:
        atoms: ASE Atoms (relaxed geometry, with PBC).
        d_a_cutoff: Donor-acceptor distance cutoff (Å).
        d_h_a_angle_cutoff: D-H-A angle cutoff (degrees).
        d_h_cutoff: Donor-H covalent bond maximum (Å).

    Returns:
        Number of H-bonds in the structure.
    """
    try:
        from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as HBA
    except ImportError:
        return count_hbonds_ase(
            atoms,
            d_a_cutoff=d_a_cutoff,
            d_h_a_angle_cutoff=d_h_a_angle_cutoff,
            d_h_cutoff=d_h_cutoff,
        )

    u = _ase_to_mda(atoms)
    hba = HBA(
        universe=u,
        donors_sel="name O",
        hydrogens_sel="name H",
        acceptors_sel="name O",
        d_a_cutoff=d_a_cutoff,
        d_h_a_angle_cutoff=d_h_a_angle_cutoff,
        d_h_cutoff=d_h_cutoff,
        update_selections=False,
    )
    hba.run(verbose=False)
    return int(len(hba.results.hbonds))


# ---------------------------------------------------------------------------
# Top-level: compose the three layers
# ---------------------------------------------------------------------------

def slab_correction(
    atoms: ase.Atoms,
    delta_O: float = DEFAULT_DELTA_O,
    eps_hbond: float = DEFAULT_EPS_HBOND,
    ads_g_intrinsic: dict[str, float] | None = None,
    metal_symbol: str = "Ir",
    return_breakdown: bool = False,
) -> float | tuple[float, dict]:
    """Total free-energy correction for a slab.

    Args:
        atoms: ASE Atoms (relaxed geometry, with PBC).
        delta_O: Per-O oxide correction (eV). See :data:`DEFAULT_DELTA_O`.
        eps_hbond: Per-H-bond correction (eV). See :data:`DEFAULT_EPS_HBOND`.
        ads_g_intrinsic: Override dict for per-adsorbate Gibbs corrections.
            Defaults to :data:`ADS_G_INTRINSIC`.
        metal_symbol: Surface metal symbol used for adsorbate identification.
        return_breakdown: If True, also return a dict of per-layer values.

    Returns:
        Total correction in eV (float) plus an optional breakdown dict.
    """
    ads_g = ads_g_intrinsic or ADS_G_INTRINSIC
    sym = atoms.get_chemical_symbols()

    # Layer A: oxide correction on ALL O atoms
    n_O = sum(1 for s in sym if s == "O")
    A = n_O * delta_O

    # Layer B: adsorbate-intrinsic Gibbs
    ads_counts = count_adsorbates(atoms, metal_symbol=metal_symbol)
    B = sum(n * ads_g.get(sp, 0.0) for sp, n in ads_counts.items())

    # Layer C: H-bonds
    n_hb = count_hbonds(atoms)
    C = n_hb * eps_hbond

    total = A + B + C
    if not return_breakdown:
        return total

    return total, {
        "A_oxide": A,
        "B_adsorbate": B,
        "C_hbond": C,
        "n_O_total": n_O,
        "ads_counts": dict(ads_counts),
        "n_hbonds": n_hb,
    }
