"""Tests for generate_pourbaix_atoms_bratsch — the pure-experimental
replacement for the MP-based pourbaix_atoms generator.

Validates:
  * Ir stable species at representative (pH, φ) grid points matches
    Bratsch 1989 phase boundaries (Ir/IrO2 at 0.725V, IrO2/IrO3 at 1.5V,
    no IrO4²⁻ at any φ ≤ 2.0V).
  * num_e, num_H, delta_G2_std fields match the Species class fields
    (which were already verified by existing bulk_pourbaix_exp tests).
  * O atom → H2O(l): ΔG₂_std = −2.458 eV, num_e=−2, num_H=−2.
  * H atom → H⁺(aq): ΔG₂_std = 0, num_e=+1, num_H=+1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_bratsch_ir_stable_species_low_phi():
    """At (pH=0, φ=0.3V), Ir bulk should be stable (< Ir/IrO2 boundary 0.725V)."""
    from mcmc.pourbaix.atoms_bratsch import generate_pourbaix_atoms_bratsch

    atoms = generate_pourbaix_atoms_bratsch(phi=0.3, pH=0.0, elements=["Ir", "O"])
    assert atoms["Ir"].dominant_species == "Ir"
    assert atoms["Ir"].num_e == 0
    assert atoms["Ir"].num_H == 0
    assert abs(atoms["Ir"].delta_G2_std) < 1e-9


def test_bratsch_ir_stable_species_mid_phi():
    """At (pH=0, φ=1.0V), IrO2(s) should be stable (0.725 < 1.0 < 1.5)."""
    from mcmc.pourbaix.atoms_bratsch import generate_pourbaix_atoms_bratsch

    atoms = generate_pourbaix_atoms_bratsch(phi=1.0, pH=0.0, elements=["Ir", "O"])
    assert atoms["Ir"].dominant_species == "IrO2"
    assert atoms["Ir"].num_e == 4
    assert atoms["Ir"].num_H == 4
    assert abs(atoms["Ir"].delta_G2_std - 2.899) < 0.01


def test_bratsch_ir_stable_species_high_phi():
    """At (pH=0, φ=1.8V), IrO3(s) should be stable (> 1.5V). NOT IrO4²⁻
    because Bratsch has no IrO4²⁻ entry (that's MP/Pourbaix-Atlas legacy)."""
    from mcmc.pourbaix.atoms_bratsch import generate_pourbaix_atoms_bratsch

    atoms = generate_pourbaix_atoms_bratsch(phi=1.8, pH=0.0, elements=["Ir", "O"])
    assert atoms["Ir"].dominant_species == "IrO3", (
        f"expected IrO3 at high φ (Bratsch), got {atoms['Ir'].dominant_species!r}. "
        "If this says IrO4[-2], the Pourbaix-Atlas estimate has been re-added "
        "to ir_species_experimental() — revert it (see bulk_pourbaix_exp.py "
        "docstring on why IrO4²⁻ is intentionally omitted)."
    )
    assert atoms["Ir"].num_e == 6
    assert atoms["Ir"].num_H == 6


def test_bratsch_oxygen_h2o_convention():
    """O atom should always resolve to H₂O(l), ΔG₂_std = -2.458, num_e=-2."""
    from mcmc.pourbaix.atoms_bratsch import generate_pourbaix_atoms_bratsch

    atoms = generate_pourbaix_atoms_bratsch(phi=0.5, pH=7.0, elements=["Ir", "O"])
    assert atoms["O"].dominant_species == "H2O"
    assert atoms["O"].num_e == -2
    assert atoms["O"].num_H == -2
    assert abs(atoms["O"].delta_G2_std - (-2.458)) < 0.01
    assert atoms["O"].species_conc == 1.0  # liquid reference


def test_bratsch_hydrogen_hplus_convention():
    """H atom should always resolve to H⁺(aq), ΔG₂_std = 0, num_e=+1."""
    from mcmc.pourbaix.atoms_bratsch import generate_pourbaix_atoms_bratsch

    atoms = generate_pourbaix_atoms_bratsch(phi=1.5, pH=0.0, elements=["Ir", "O"])
    assert atoms["H"].dominant_species == "H[+1]"
    assert atoms["H"].num_e == 1
    assert atoms["H"].num_H == 1
    assert atoms["H"].delta_G2_std == 0.0
    assert atoms["H"].species_conc == 1.0


def test_bratsch_basic_high_phi_still_iro3():
    """pH=14, φ=2.0V: still IrO3, no IrO4²⁻ artifact from MP."""
    from mcmc.pourbaix.atoms_bratsch import generate_pourbaix_atoms_bratsch

    atoms = generate_pourbaix_atoms_bratsch(phi=2.0, pH=14.0, elements=["Ir", "O"])
    assert atoms["Ir"].dominant_species == "IrO3"


def test_bratsch_no_registered_metal_raises():
    """Metals without a curated species list should raise, not silently pick the wrong thing."""
    from mcmc.pourbaix.atoms_bratsch import generate_pourbaix_atoms_bratsch

    with pytest.raises(NotImplementedError, match="Bratsch"):
        generate_pourbaix_atoms_bratsch(phi=1.0, pH=0.0, elements=["Pt", "O"])
