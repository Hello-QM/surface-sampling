"""Pure-experimental generator for ``dict[element → PourbaixAtom]`` used by
``HamiltonianREMC`` during MC sampling.

Replaces ``generate_pourbaix_atoms()`` which goes through the MP
(``pymatgen.analysis.pourbaix_diagram.PourbaixDiagram.get_stable_entry``)
path — that path pulls in the Pourbaix Atlas 1974 "approximate" IrO₄²⁻
value and the MP DFT+MP2020 IrO₂ value, both of which the project has
explicitly flagged as legacy.

This generator uses **directly-measured experimental data only**:
  * Bratsch 1989 (J. Phys. Chem. Ref. Data 18, 1) for E° of Ir/IrO₂,
    IrO₂/IrO₃, Ir/Ir³⁺ reductions.
  * Cordfunke & Konings 1981 for ΔG°_f(IrO₂,s) = −2.017 eV/fu.
  * Experimental anchor ΔG°_f(H₂O,l) = −2.458 eV/fu.
  * H⁺(aq) convention: ΔG°_f = 0 at 1 M, pH=0.

The species list comes from ``bulk_pourbaix_exp.ir_species_experimental``
which the user curated in /home/james0001/project/pourbaix-diagram/.

Absent from this path:
  * IrO₄²⁻(aq) — Bratsch has no verified value; the MP/Pourbaix-Atlas
    estimate would introduce legacy data. IrO₃(s) covers the high-φ
    region for OER-relevant (pH, φ) ≤ (14, 2.0 V).

Use via the --pourbaix_source bratsch flag on run_hre_pourbaix_3layer.py;
this is now the default so MC sampling no longer needs the
--experimental_override IrO2:+1.76 hack.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from mcmc.pourbaix.atoms import PourbaixAtom

logger = logging.getLogger(__name__)


def _import_bulk_pourbaix_exp():
    """Locate and import bulk_pourbaix_exp.

    The module lives at the project root /home/james0001/project/pourbaix-diagram/
    (local) or /expanse/projects/qstore/csd807/gliu3/containers/ (Expanse).
    Both are outside the surface-sampling repo, so we probe a couple of
    canonical locations and inject them into sys.path.
    """
    for candidate in (
        Path("/home/james0001/project/pourbaix-diagram"),
        Path("/expanse/projects/qstore/csd807/gliu3/containers"),
    ):
        if (candidate / "bulk_pourbaix_exp.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break
    try:
        import bulk_pourbaix_exp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "bulk_pourbaix_exp could not be imported. Tried "
            "/home/james0001/project/pourbaix-diagram/ and "
            "/expanse/projects/qstore/csd807/gliu3/containers/. "
            f"Underlying error: {exc}"
        )
    return bulk_pourbaix_exp


# Default metal-species registry: map element symbol → callable that returns
# a list[bulk_pourbaix_exp.Species] of that element's experimental Pourbaix
# species. Add new metals (Ru, Mn, Sr …) here as their species lists are
# curated; until then, sampling non-Ir systems falls through to an
# informative NotImplementedError.
def _metal_species_registry():
    bpe = _import_bulk_pourbaix_exp()
    return {
        "Ir": bpe.ir_species_experimental,
    }


def generate_pourbaix_atoms_bratsch(
    phi: float,
    pH: float,
    elements: list[str],
    metal_species_overrides: dict[str, list] | None = None,
) -> dict[str, "PourbaixAtom"]:
    """Build a per-element ``PourbaixAtom`` dict using Bratsch 1989 + Cordfunke
    1981 experimental data (no MP / Pourbaix Atlas estimates).

    Args:
        phi: Electrode potential (V vs SHE).
        pH: Bulk pH.
        elements: Metal/non-metal element symbols present in the slab,
            e.g. ``["Ir", "O"]``. O and H are handled via the standard
            H₂O(l) / H⁺(aq) conventions regardless of whether they appear
            in ``elements``.
        metal_species_overrides: Optional dict overriding the default
            metal → species-list mapping. Useful for multi-metal systems
            where the user wants to plug in their own ``ru_species_*`` or
            ``mn_species_*`` list without editing the registry.

    Returns:
        dict mapping element symbol → PourbaixAtom ready to be consumed
        by ``MACEPourbaix.set(pourbaix_atoms=...)`` during MC.
    """
    bpe = _import_bulk_pourbaix_exp()
    registry = _metal_species_registry()
    if metal_species_overrides:
        registry.update(metal_species_overrides)

    atoms: dict[str, PourbaixAtom] = {}

    # Metal atoms — pick the stable species at (pH, φ) from the experimental
    # list and wrap it in a PourbaixAtom whose num_e / num_H / dG2_std match
    # the oxidation half-reaction α M + β H₂O → species + γ H⁺ + δ e⁻.
    for elem in elements:
        if elem in ("O", "H"):
            continue
        if elem not in registry:
            raise NotImplementedError(
                f"No Bratsch / experimental species list registered for {elem!r}. "
                "Add a callable to _metal_species_registry or pass via "
                "metal_species_overrides=..."
            )
        species_list = registry[elem]()
        # Find stable species (min Gibbs energy per metal atom at (pH, φ))
        stable = min(species_list, key=lambda s: s.G(pH, phi))
        atoms[elem] = PourbaixAtom(
            symbol=elem,
            dominant_species=stable.name,
            species_conc=stable.activity,
            num_e=int(round(stable.n_e)),
            num_H=int(round(stable.n_H_released)),
            atom_std_state_energy=0.0,
            delta_G2_std=float(stable.dG2_std),
        )
        logger.debug(
            "Bratsch Pourbaix atom @ pH=%.1f, φ=%.2f: %s → %s "
            "(n_e=%+d, n_H=%+d, ΔG₂_std=%+.4f eV, a=%.0e)",
            pH, phi, elem, stable.name,
            int(stable.n_e), int(stable.n_H_released),
            stable.dG2_std, stable.activity,
        )

    # Oxygen → H₂O(l) reduction half-reaction (per O atom):
    #   O_std + 2 H⁺(aq) + 2 e⁻ → H₂O(l)
    # Equivalently the oxidation-side form used by MACEPourbaix with
    # delta_G2_non_std = -n_e·φ - ln10·n_H·kT·pH + kT·ln(a).
    # For an O atom going to H₂O: n_e = −2, n_H = −2 (H⁺ and e⁻ are
    # consumed, not produced), delta_G2_std = ΔG°_f(H₂O,l) = −2.458 eV.
    atoms["O"] = PourbaixAtom(
        symbol="O",
        dominant_species="H2O",
        species_conc=1.0,               # pure liquid reference (a = 1)
        num_e=-2,
        num_H=-2,
        atom_std_state_energy=0.0,
        delta_G2_std=float(bpe.DG_F_H2O),  # -2.458 eV
    )

    # Hydrogen → H⁺(aq) oxidation half-reaction (per H atom):
    #   ½ H₂ → H⁺(aq) + e⁻
    # Standard state by convention: ΔG°_f(H⁺, 1 M) = 0.
    atoms["H"] = PourbaixAtom(
        symbol="H",
        dominant_species="H[+1]",
        species_conc=1.0,               # 1 M reference (conc_term = 0)
        num_e=1,
        num_H=1,
        atom_std_state_energy=0.0,
        delta_G2_std=0.0,
    )

    return atoms
