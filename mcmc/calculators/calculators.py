"""Module for ASE-style Calculators for surface energy calculations."""

import json
import logging
import os
from collections import Counter

import ase
import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.formula import Formula
try:
    from lammps import (
        LMP_STYLE_ATOM,
        LMP_STYLE_GLOBAL,
        LMP_TYPE_SCALAR,
        LMP_TYPE_VECTOR,
        lammps,
    )
except ImportError:
    LMP_STYLE_ATOM = LMP_STYLE_GLOBAL = LMP_TYPE_SCALAR = LMP_TYPE_VECTOR = None
    lammps = None

from mace.calculators import MACECalculator

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

try:
    from mcmc.pourbaix.atoms import PourbaixAtom
except ImportError:
    PourbaixAtom = None

try:
    from .lammpsrun import LAMMPS as LAMMPSRun
except ImportError:
    LAMMPSRun = None

logger = logging.getLogger(__name__)

ENERGY_THRESHOLD = 1000  # eV
MAX_FORCE_THRESHOLD = 1000  # eV/Angstrom
HARTREE_TO_EV = 27.211386245988


def get_results_single(atoms: Atoms, calc: Calculator) -> dict:
    """Calculate the results for a single Atoms object.

    Args:
        atoms (Atoms): The Atoms object to calculate the results for.
        calc (Calculator): The calculator to use.

    Returns:
        dict: The results of the calculation
    """
    atoms.calc = calc
    calc.calculate(atoms)

    return calc.results


def get_embeddings(atoms_list: list[Atoms], calc: Calculator) -> np.ndarray:
    """Calculate the embeddings for a list of Atoms objects

    Args:
        atoms_list (list[Atoms]): List of Atoms objects.
        calc (Calculator): MACE Calculator.

    Returns:
        np.ndarray: Latent space embeddings with each row corresponding to a structure
    """
    print(f"Calculating embeddings for {len(atoms_list)} structures")
    embeddings = []
    for atoms in tqdm(atoms_list):
        embedding = get_embeddings_single(atoms, calc)
        embeddings.append(embedding)
    return np.stack(embeddings)


def get_embeddings_single(
    atoms: Atoms,
    calc: Calculator,
    results_cache: dict | None = None,
    flatten: bool = True,
    flatten_axis: int = 0,
) -> np.ndarray:
    """Calculate the embeddings for a single Atoms object using MACE forward hooks.

    Args:
        atoms (Atoms): Atoms object.
        calc (Calculator): MACE Calculator (MACESurface or MACECalculator).
        results_cache (dict): Cache for results (unused, kept for API compatibility).
        flatten (bool): Whether to flatten the embeddings.
        flatten_axis (int): Axis to flatten the embeddings.

    Returns:
        np.ndarray: Latent space embeddings
    """
    import torch

    # Get the underlying MACE model
    if hasattr(calc, "calculators"):
        mace_calc = calc.calculators[0]
    elif hasattr(calc, "mace_calc"):
        mace_calc = calc.mace_calc
    else:
        mace_calc = calc

    model = mace_calc.models[0]
    embeddings = []

    def hook_fn(mod, inp, out):
        if isinstance(out, tuple):
            embeddings.append(out[0].detach().cpu())
        else:
            embeddings.append(out.detach().cpu())

    hook = model.products[-1].register_forward_hook(hook_fn)
    try:
        mace_calc.calculate(atoms)
    finally:
        hook.remove()

    if embeddings:
        emb = embeddings[0]  # (num_atoms, embedding_dim)
        emb_np = emb.numpy()
        if flatten:
            return emb_np.mean(axis=flatten_axis).squeeze()
        return emb_np.squeeze()

    # Fallback: return zeros if hook didn't fire
    mace_calc.calculate(atoms)
    return np.zeros(128)


def get_std_devs(atoms_list: list[Atoms], calc: Calculator) -> np.ndarray:
    """Calculate the force standard deviations across multiple models for a list of Atoms objects

    Args:
        atoms_list (List[Atoms]): List of Atoms objects
        calc (Calculator): MACE Calculator

    Returns:
        np.ndarray: Force standard deviation with a single value for each structure
    """
    print(f"Calculating force standard deviations for {len(atoms_list)} structures")
    force_stds = []
    for atoms in tqdm(atoms_list):
        force_std = get_std_devs_single(atoms, calc)
        force_stds.append(force_std)

    return np.stack(force_stds)


def get_std_devs_single(atoms: Atoms, calc: Calculator) -> float:
    """Calculate the force standard deviation for a single Atoms object

    Args:
        atoms (Atoms): Atoms object
        calc (Calculator): MACE Calculator (MACESurface with ensemble)

    Returns:
        float: Force standard deviation
    """
    if hasattr(calc, "calculators") and len(calc.calculators) > 1:
        forces_list = []
        for c in calc.calculators:
            c.calculate(atoms)
            forces_list.append(c.results["forces"].copy())
        return np.std(forces_list, axis=0).mean()
    return 0.0


class MACESurface(Calculator):
    """MACE-based surface energy calculator with optional ensemble."""

    implemented_properties = ("energy", "forces", "stress", "surface_energy")

    def __init__(self, model_paths, device="cuda", enable_cueq=True, **kwargs):
        """Initialize the MACESurface class.

        Args:
            model_paths: List of paths to MACE model files.
            device: Device to use ('cuda' or 'cpu').
            enable_cueq: Whether to enable cuEquivariance.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        self.calculators = [
            MACECalculator(model_paths=p, device=device, enable_cueq=enable_cueq)
            for p in model_paths
        ]
        self.chem_pots = {}
        self.offset_data = {}
        self.offset_units = kwargs.get("offset_units", "eV")
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

    def get_surface_energy(
        self,
        atoms: ase.Atoms = None,
        chem_pots: dict | None = None,
        offset_data: dict | None = None,
    ) -> float:
        """Get the surface energy of the system by subtracting the bulk energy and the chemical
        potential deviation from the bulk formula. Refer to Methods-Surface stability analysis
        section of Du, X. et al. Nat Comput Sci 1-11 (2023) doi:10.1038/s43588-023-00571-7
        for details.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the surface energy for.
            chem_pots (dict): The chemical potentials of the atoms in the system.
            offset_data (dict): The offset data for the system.

        Returns:
            float: The surface energy of the system.
        """
        if atoms is None:
            atoms = self.atoms

        if chem_pots is None:
            chem_pots = self.chem_pots

        if offset_data is None:
            offset_data = self.offset_data
        if not offset_data:
            self.logger.debug("No offset_data, using raw potential energy as surface energy")
            return self.get_potential_energy(atoms=atoms)

        # Starting with the potential energy of the system (akin to DFT energy of the slab)
        surface_energy = self.get_potential_energy(atoms=atoms)

        ads_count = Counter(atoms.get_chemical_symbols())

        bulk_energies = offset_data["bulk_energies"]
        stoics = offset_data["stoics"]
        ref_formula = offset_data["ref_formula"]
        ref_element = offset_data["ref_element"]

        # Subtract the bulk energies
        bulk_ref_en = ads_count[ref_element] * bulk_energies[ref_formula]
        for ele in ads_count:
            if ele != ref_element:
                bulk_ref_en += (
                    ads_count[ele] - stoics.get(ele, 0) / stoics[ref_element] * ads_count[ref_element]
                ) * bulk_energies[ele]

        if self.offset_units == "atomic":
            surface_energy -= bulk_ref_en * HARTREE_TO_EV
        else:
            surface_energy -= bulk_ref_en

        # Subtract chemical potential deviation from bulk formula
        stoics = self.offset_data["stoics"]
        ref_element = self.offset_data["ref_element"]

        pot = 0
        for ele in ads_count:
            if ele != ref_element:
                pot += (
                    ads_count[ele] - stoics.get(ele, 0) / stoics[ref_element] * ads_count[ref_element]
                ) * self.chem_pots[ele]

        surface_energy -= pot
        return surface_energy

    def set(self, **kwargs) -> dict:
        """Set parameters in key-value pairs. A dictionary containing the parameters that have been
        changed is returned. The special keyword 'parameters' can be used to read parameters from a
        file.

        Args:
            **kwargs: The parameters to set.

        Returns:
            dict: A dictionary containing the parameters that have been changed.
        """
        changed_parameters = Calculator.set(self, **kwargs)
        if "chem_pots" in self.parameters:
            self.chem_pots = self.parameters["chem_pots"]
            self.logger.info("chemical potentials: %s are set from parameters", self.chem_pots)
        if "offset_data" in self.parameters:
            self.offset_data = self.parameters["offset_data"]
            self.logger.info("offset data: %s is set from parameters", self.offset_data)
        return changed_parameters

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties: tuple = ("energy", "forces"),
        system_changes: list = all_changes,
    ):
        """Calculate using MACE ensemble before adding surface energy calcs to results.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the properties for.
            properties (tuple): The properties to calculate.
            system_changes (list): The system changes to calculate.
        """
        if atoms is None:
            atoms = self.atoms

        Calculator.calculate(self, atoms, properties, system_changes)

        # Use first calculator for primary results
        self.calculators[0].calculate(atoms)
        self.results.update(self.calculators[0].results)

        # Ensemble: compute forces_std from all models
        if len(self.calculators) > 1:
            all_forces = [self.calculators[0].results["forces"].copy()]
            for calc in self.calculators[1:]:
                calc.calculate(atoms)
                all_forces.append(calc.results["forces"].copy())
            self.results["forces_std"] = np.std(all_forces, axis=0).mean()

        if "surface_energy" in properties:
            self.results["surface_energy"] = self.get_surface_energy(atoms=atoms)


class MACEPourbaix(Calculator):
    """Calculate Pourbaix potential for surface or bulk systems using MACE.
    We calculate the energy difference based on consecutive desorption/adsorption reactions as in
    Rong and Kolpak, J. Phys. Chem. Lett., 2015.
    Each step consists of the following:
    1. G_ref -> G_new + A
    2. A + n_H2O H2O -> HxAOy^(z-) + n_H+ H+ + n_e e-
    Delta G_overall = Delta G_1 + Delta G_2
    The free energy change for the first step is given by:
    Delta G_1 = G_new + mu_A - G_ref
    where G_new is the energy of the new system, mu_A is the chemical potential of the element A,
    and G_ref is the energy of the reference system.
    The free energy change for the second step is given by:
    Delta G_2 = Delta G_SHE - n_e (e*U_SHE) - 2.3 n_H+ k_B T pH + k_B T ln(a_HxAOy^(z-))
    where Delta G_SHE is the energy change at standard hydrogen electrode potential,
    e is the electron charge, U_SHE is the standard hydrogen electrode potential,
    n_H+ is the number of protons, k_B is the Boltzmann constant, T is the temperature,
    pH is the pH, and a_HxAOy^(z-) is the activity of the species.

    Attributes:
        implemented_properties (list): List of implemented properties.
        chem_pots (dict): Dictionary of chemical potentials.
        reference_slab (dict): Dictionary of reference slabs.
        temp (float): Temperature in eV.
        phi (float): Electric potential.
        pH (float): pH value.
        pourbaix_atoms (dict): Dictionary of pourbaix atoms.

    Methods:
        get_delta_G2_individual: Get the standard free energy change for the second step of the
            Pourbaix reaction for a single atom.
        get_delta_G2: Get the standard free energy change for the second step of the
            Pourbaix reaction.
        get_delta_G1: Get the dissociation energy of all atoms.
        get_surface_energy: Get the surface energy of the system, which is equivalent to the
            Pourbaix potential.
        get_pourbaix_potential: Get the Pourbaix potential of the system.
        set: Set parameters.
        calculate: Calculate based on MACE before adding surface energy calculations to results.
    """

    implemented_properties = (
        "energy",
        "forces",
        "stress",
        "pourbaix_potential",
        "surface_energy",
    )

    def __init__(self, model_path, device="cuda", enable_cueq=True, **kwargs):
        """Initialize the MACEPourbaix class.

        Args:
            model_path: Path to MACE model file (str) or list with single path.
            device: Device to use.
            enable_cueq: Whether to enable cuEquivariance.
            **kwargs: Additional keyword arguments including:
                offset_data (dict): MACE-consistent bulk reference energies for
                    surface energy computation. Keys: bulk_energies, stoics,
                    ref_formula, ref_element. When provided, get_delta_G1 uses
                    MACE bulk energies instead of MP atom_std_state_energy,
                    ensuring slab and reference are on the same energy scale.
        """
        super().__init__(**kwargs)
        if isinstance(model_path, list):
            model_path = model_path[0]
        self.mace_calc = MACECalculator(
            model_paths=model_path, device=device, enable_cueq=enable_cueq
        )
        self.chem_pots = {}
        self.reference_slab = {}
        self.temp = kwargs.get("temp", 0.0257)  # temperature in eV
        self.phi = kwargs.get("phi", 0)  # electric potential
        self.pH = kwargs.get("pH", 7)  # pH
        self.pourbaix_atoms = {}
        self.offset_data = kwargs.get("offset_data", {})
        # LEGACY: per-adsorbate single-value correction dict (e.g. {"HO": 0.23}).
        # Kept for backward compatibility / comparison; only used when
        # use_adsorbate_gibbs is explicitly set to False. New workflows
        # should leave use_adsorbate_gibbs at the default (True) and configure
        # slab_correction_kwargs instead.
        self.adsorbate_corrections = {}
        # AUTHORITATIVE PATH — 3-layer slab_correction (default = True).
        # When True, get_delta_G1 calls
        # mcmc.corrections.adsorbate_gibbs.slab_correction(atoms, **kwargs)
        # to add:
        #   Layer A — per-oxide Δ_O (fit to experimental ΔG°_f; default
        #             -0.1252 eV/O for IrO₂, configurable via
        #             slab_correction_kwargs["delta_O"]).
        #   Layer B — per-species adsorbate Gibbs (ZPE + ∫Cp dT − TS) for
        #             *O/*OH/*OOH/*H₂O/*H, with geometric identification by
        #             Ir-coordination (metal_symbol configurable).
        #   Layer C — hydrogen-bond count × ε_HB (Luzar-Chandler criterion
        #             via MDAnalysis).
        # Replaces both the legacy MP2020 oxide_correction_per_O (Layer A)
        # and the self.adsorbate_corrections dict (Layer B). See
        # mcmc/corrections/adsorbate_gibbs.py for the scheme and
        # learning_notes.md §19 for the thermodynamic derivation.
        self.use_adsorbate_gibbs = True
        self.slab_correction_kwargs = {}
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

    def get_delta_G2_individual(self, atom: str | PourbaixAtom) -> float:
        """Get the free energy change for the second step of the Pourbaix reaction for a
        single atom.

        Args:
            atom (Union[str, PourbaixAtom]): The atom to calculate the free energy change for.

        Returns:
            float: The standard free energy change for the second step for a single atoms.
        """
        if isinstance(atom, str):
            atom = self.pourbaix_atoms[atom]
        # - n_e (e*U_SHE) - 2.3 n_H+ k_B T pH + k_B T ln(a_HxAOy^(z-))
        delta_G2_non_std = (
            -atom.num_e * self.phi
            - np.log(10) * atom.num_H * self.temp * self.pH
            + self.temp * np.log(atom.species_conc)
        )
        return atom.delta_G2_std + delta_G2_non_std

    def get_delta_G2(self, atoms: ase.Atoms = None) -> float:
        """Get the total free energy change for the second step of the Pourbaix reaction.

        Args:
            atoms (ase.Atoms, optional): The atoms object to calculate the free energy change for.
                Defaults to None.

        Returns:
            float: The total standard free energy change for the second step.
        """
        if atoms is None:
            atoms = self.atoms

        delta_G2 = 0
        for atom in atoms.get_chemical_symbols():
            delta_G2 += self.get_delta_G2_individual(atom)
        return delta_G2

    def get_delta_G1(self, atoms: ase.Atoms = None) -> float:
        """Get the dissociation energy of all atoms.

        When offset_data is provided, uses MACE-consistent bulk reference energies
        (same energy scale as the MACE slab energy) instead of Materials Project
        atom_std_state_energy values. This eliminates the energy reference mismatch
        between MACE slab energies and MP/DFT chemical potentials.

        The surface energy convention with offset_data:
            delta_G1 = E_slab - E_bulk_ref
        where E_bulk_ref is computed from MACE bulk energies using the same
        stoichiometric subtraction as MACESurface.get_surface_energy().

        If ``oxide_correction_per_O`` is present in offset_data, the
        MP2020Compatibility oxide anion correction is applied to bring
        MACE energies (trained on uncorrected PBE) onto the same scale
        as the MP Pourbaix ion data used in delta_G2. The correction is
        added to both the slab energy and the bulk reference energies
        (already baked into the corrected bulk_energies). See
        ``apply_compatibility.py`` for derivation.

        Without offset_data (backward compatible):
            delta_G1 = sum(mu_std) - E_slab

        Args:
            atoms (ase.Atoms, optional): The atoms object to calculate the dissociation energy for.
                Defaults to None.

        Returns:
            float: The dissociation energy of all atoms.
        """
        if atoms is None:
            atoms = self.atoms

        slab_energy = self.get_potential_energy(atoms=atoms)

        # 3-layer slab correction (opt-in). Added to slab_energy so the
        # downstream ΔG₁ formula absorbs it consistently:
        #   non-offset:  ΔG₁ = Σ μ° − (E_NFF + G_corr)
        #   offset:      ΔG₁ = (E_NFF + G_corr) − E_bulk_ref
        # Replaces both the legacy MP2020 oxide_correction_per_O (Layer A) and
        # the self.adsorbate_corrections dict loop (Layer B). See
        # mcmc/corrections/adsorbate_gibbs.py for the experimental-first
        # per-oxide Δ_O + per-species ZPE-TS + MDAnalysis H-bond count.
        if self.use_adsorbate_gibbs:
            from mcmc.corrections.adsorbate_gibbs import slab_correction
            G_corr = slab_correction(atoms, **self.slab_correction_kwargs)
            slab_energy = slab_energy + G_corr

        if self.offset_data:
            # Use MACE-consistent bulk references (same energy scale as slab)
            ads_count = Counter(atoms.get_chemical_symbols())
            bulk_energies = self.offset_data["bulk_energies"]
            stoics = self.offset_data["stoics"]
            ref_formula = self.offset_data["ref_formula"]
            ref_element = self.offset_data["ref_element"]

            # Apply MP2020 oxide anion correction to slab energy if available.
            # MACE predicts on the uncorrected DFT scale; the Pourbaix ion data
            # (delta_G2_std) is calibrated against MP-corrected DFT. Adding the
            # oxide correction here brings delta_G1 onto the same scale.
            # The bulk_energies in offset_data should already include the
            # correction for compounds (IrO2, H2O) but not for elemental refs.
            # Skipped when use_adsorbate_gibbs is True — Layer A of the 3-layer
            # scheme already applies a per-oxide Δ_O (fit to experimental
            # ΔG°_f) which supersedes MP2020's generic -0.687 eV/O.
            if not self.use_adsorbate_gibbs:
                oxide_corr = self.offset_data.get("oxide_correction_per_O", 0.0)
                if oxide_corr != 0.0:
                    n_O_slab = ads_count.get("O", 0)
                    slab_energy += oxide_corr * n_O_slab

            bulk_ref_en = ads_count[ref_element] * bulk_energies[ref_formula]
            for ele in ads_count:
                if ele != ref_element:
                    bulk_ref_en += (
                        ads_count[ele] - stoics.get(ele, 0) / stoics[ref_element] * ads_count[ref_element]
                    ) * bulk_energies[ele]

            # Surface energy convention: no negative sign
            delta_G1 = slab_energy - bulk_ref_en
        else:
            # Original: uses atom_std_state_energy from PhaseDiagram (MP DFT)
            atoms_count = Counter(atoms.get_chemical_symbols())
            sum_chem_pots = 0
            for atom, count in atoms_count.items():
                sum_chem_pots += count * self.pourbaix_atoms[atom].atom_std_state_energy
            delta_G1 = sum_chem_pots - slab_energy

        # Add adsorbate corrections, e.g. OH ZPE-TS correction
        # Note: corrections are only applied if the adsorbate species is actually
        # present in the slab formula. If not, divmod returns 0 and no correction
        # is added.
        # Skipped when use_adsorbate_gibbs is True — Layer B of the 3-layer
        # scheme already applies per-species ZPE-TS (for *O, *OH, *OOH, *H₂O,
        # *H) via geometric adsorbate identification, and Layer C counts
        # hydrogen bonds via MDAnalysis. These supersede the divmod-by-formula
        # scheme here (which only knew about OH).
        formula = Formula(atoms.get_chemical_formula())
        for adsorbate, correction in (
            {} if self.use_adsorbate_gibbs else self.adsorbate_corrections
        ).items():
            # Check for H2O
            if "O" in adsorbate and "H" in adsorbate:
                # Assume the extra H is from water so subtract H2O from the formula
                HO_diff = max(formula["H"] - formula["O"], 0)
                if HO_diff > 0:
                    logger.info("Correcting formula %s with HO diff %s", formula, HO_diff)
                    formula_dict_to_subtract = (Formula("H2O") * HO_diff).count()
                    formula_dict = formula.count()
                    formula_dict = {
                        formula: formula_dict[formula] - formula_dict_to_subtract.get(formula, 0)
                        for formula in formula_dict
                    }
                    formula = Formula.from_dict(formula_dict)
                    logger.info("Corrected formula %s", formula)
            div, _ = divmod(formula, adsorbate)
            delta_G1 += div * correction
        return delta_G1

    def get_surface_energy(self, atoms: ase.Atoms = None) -> float:
        """Get the surface energy of the system, which is equivalent to the Pourbaix potential.
        See get_pourbaix_potential for more details.

        Args:
            atoms (ase.Atoms, optional): The atoms object to calculate the surface energy for.
                Defaults to None.

        Returns:
            float: The surface energy of the system.
        """
        if atoms is None:
            atoms = self.atoms

        return self.get_pourbaix_potential(atoms=atoms)

    def get_pourbaix_potential(self, atoms: ase.Atoms = None) -> float:
        """Get the Pourbaix potential of the system, which is the negative of the sum of the free
        energy changes for the two steps of the Pourbaix dissolution reaction. This is also the
        "surface free energy" and the "Grand potential" in the Pourbaix diagram.

        Args:
            atoms (ase.Atoms, optional): The atoms object to calculate the Pourbaix potential for.
                Defaults to None.

        Returns:
            float: The Pourbaix potential of the system.
        """
        if atoms is None:
            atoms = self.atoms

        return -(self.get_delta_G1(atoms=atoms) + self.get_delta_G2(atoms=atoms))

    def set(self, **kwargs) -> dict:
        """Set parameters in key-value pairs. A dictionary containing the parameters that have been
        changed is returned. The special keyword 'parameters' can be used to read parameters from a
        file.

        Args:
            **kwargs: The parameters to set.

        Returns:
            dict: A dictionary containing the parameters that have been changed.
        """
        changed_params = Calculator.set(self, **kwargs)
        if "temperature" in self.parameters:
            self.temp = self.parameters["temperature"]
            self.logger.info("temperature: %.3f in kBT", self.temp)
        if "phi" in self.parameters:
            self.phi = self.parameters["phi"]
            self.logger.info("potential: %.3f is set from parameters", self.phi)
        if "pH" in self.parameters:
            self.pH = self.parameters["pH"]
            self.logger.info("pH: %.3f is set from parameters", self.pH)
        if "pourbaix_atoms" in self.parameters:
            self.pourbaix_atoms = self.parameters["pourbaix_atoms"]
            self.logger.info("Pourbaix atoms: %s are set from parameters", self.pourbaix_atoms)
        if "offset_data" in self.parameters:
            self.offset_data = self.parameters["offset_data"]
            self.logger.info("offset data: %s is set from parameters", self.offset_data)
        if "adsorbate_corrections" in self.parameters:
            self.adsorbate_corrections = self.parameters["adsorbate_corrections"]
            self.logger.info(
                "adsorbate corrections: %s are set from parameters", self.adsorbate_corrections
            )
        if "use_adsorbate_gibbs" in self.parameters:
            self.use_adsorbate_gibbs = bool(self.parameters["use_adsorbate_gibbs"])
            if self.use_adsorbate_gibbs:
                self.logger.info(
                    "ΔG_pbx path: 3-layer slab_correction (authoritative)"
                )
            else:
                self.logger.warning(
                    "ΔG_pbx path: LEGACY adsorbate_corrections dict + MP2020 "
                    "oxide_correction_per_O. The 3-layer scheme "
                    "(use_adsorbate_gibbs=True, default) is authoritative; "
                    "use legacy only for comparison / reproducing old results."
                )
        if "slab_correction_kwargs" in self.parameters:
            self.slab_correction_kwargs = self.parameters["slab_correction_kwargs"]
            self.logger.info(
                "slab_correction_kwargs: %s", self.slab_correction_kwargs
            )
        return changed_params

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties: tuple = ("energy", "forces"),
        system_changes: list = all_changes,
    ):
        """Calculate based on MACE before adding surface energy calcs to results.
        Args:
            atoms: ase.Atoms
                The atoms object to calculate the properties for.
            properties: List
                The properties to calculate.
            system_changes: List
                The system changes to calculate.
        """
        if atoms is None:
            atoms = self.atoms

        Calculator.calculate(self, atoms, properties, system_changes)

        self.mace_calc.calculate(atoms)
        self.results.update(self.mace_calc.results)

        if "surface_energy" in properties:
            self.results["surface_energy"] = self.get_pourbaix_potential(atoms=atoms)


# Backward-compatible aliases
EnsembleNFFSurface = MACESurface
NFFPourbaix = MACEPourbaix


class LAMMMPSCalc(Calculator):
    """Custom LAMMPSCalc class to calculate energies and forces to inteface with ASE."""

    name = "lammpscalc"
    implemented_properties = ("energy", "relaxed_energy", "forces", "per_atom_energies")
    # NOTE "energy" is the unrelaxed energy

    def __init__(self, *args, **kwargs):
        """Initialize the LAMMMPSCalc class."""
        super().__init__(*args, **kwargs)
        self.run_dir = os.getcwd()
        self.relax_steps = 100
        self.kim_potential = False
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

    def run_lammps_calc(
        self,
        slab,
        run_dir="./",
        template_path="./lammps_opt_template.txt",
        lammps_config="./lammps_config.json",
        **kwargs,
    ) -> tuple:
        """Main function to run LAMMPS calculation. Can be used for both relaxation and static
        energy calculations.

        Args:
            slab (ase.Atoms): The slab to calculate the energy for.
            run_dir (str): The directory to run LAMMPS in.
            template_path (str): The path to the LAMMPS input template file.
            lammps_config (str): The path to the LAMMPS config file.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: The slab with the new atomic positions, the energy, and the per atom energies.
        """
        with open(template_path, "r", encoding="utf-8") as f:
            lammps_template = f.read()

        # config file is assumed to be stored in the folder you run lammps
        if isinstance(lammps_config, str):
            with open(lammps_config, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = lammps_config

        potential_file = config.get("potential_file")
        atoms = config["atoms"]
        bulk_index = config["bulk_index"]

        # define necessary file locations
        lammps_data_file = f"{run_dir}/lammps.data"
        lammps_in_file = f"{run_dir}/lammps.in"
        lammps_out_file = f"{run_dir}/lammps.out"

        # write current surface into lammps.data
        slab.write(lammps_data_file, format="lammps-data", units="real", atom_style="atomic")
        steps = kwargs.get("relax_steps", 100)

        # write lammps.in file
        with open(lammps_in_file, "w", encoding="utf-8") as f:
            # if using KIM potential
            if kwargs.get("kim_potential", False):
                f.writelines(
                    lammps_template.format(lammps_data_file, bulk_index, steps, lammps_out_file)
                )
            else:
                f.writelines(
                    lammps_template.format(
                        lammps_data_file,
                        bulk_index,
                        potential_file,
                        *atoms,
                        steps,
                        lammps_out_file,
                    )
                )

        # run LAMMPS without too much output
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
        # lmp = lammps()
        self.logger.info(lmp.file(lammps_in_file))

        energy = lmp.extract_compute("thermo_pe", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR)
        if "opt" in lammps_template:
            pe_per_atom = []
        else:
            pe_per_atom = lmp.extract_compute("pe_per_atom", LMP_STYLE_ATOM, LMP_TYPE_VECTOR)
            pe_per_atom = np.ctypeslib.as_array(
                pe_per_atom, shape=(len(slab),)
            )  # convert to numpy array
        lmp.close()

        # Read from LAMMPS out
        new_slab = ase.io.read(lammps_out_file, format="lammps-data", style="atomic")

        atomic_numbers_dict = config["atomic_numbers_dict"]

        # For some ase versions, the retrieved atomic numbers are not the 'real' atomic numbers
        if not set(new_slab.get_atomic_numbers()) <= set(atomic_numbers_dict.values()):
            actual_atomic_numbers = [
                atomic_numbers_dict[str(x)] for x in new_slab.get_atomic_numbers()
            ]
            new_slab.set_atomic_numbers(actual_atomic_numbers)
        new_slab.calc = slab.calc

        return energy, pe_per_atom, new_slab

    def run_lammps_opt(self, slab, run_dir="./", **kwargs) -> tuple:
        """Run LAMMPS relaxation calculation.

        Args:
            slab (ase.Atoms): The slab to calculate the energy for.
            run_dir (str): The directory to run LAMMPS in.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            tuple: The slab with the new atomic positions, the energy, and the per atom energies.
        """
        energy, pe_per_atom, opt_slab = self.run_lammps_calc(
            slab,
            run_dir=run_dir,
            template_path=f"{run_dir}/lammps_opt_template.txt",
            lammps_config=f"{run_dir}/lammps_config.json",
            **kwargs,
        )
        self.logger.debug("slab energy in relaxation: %.3f", energy)
        return opt_slab, energy, pe_per_atom

    def run_lammps_energy(self, slab, run_dir="./", **kwargs) -> tuple:
        """Run LAMMPS static energy calculation.

        Args:
            slab (ase.Atoms): The slab to calculate the energy for.
            run_dir (str): The directory to run LAMMPS in.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            tuple: The slab with the new atomic positions, the energy, and the per atom energies.
        """
        energy, pe_per_atom, _ = self.run_lammps_calc(
            slab,
            run_dir=run_dir,
            template_path=f"{run_dir}/lammps_energy_template.txt",
            lammps_config=f"{run_dir}/lammps_config.json",
            **kwargs,
        )
        self.logger.debug("slab energy in engrad: %.3f", energy)
        return slab, energy, pe_per_atom

    def set(self, **kwargs) -> dict:
        """Set parameters in key-value pairs. A dictionary containing the parameters that have been
        changed is returned. The special keyword 'parameters' can be used to read parameters from a
        file.

        Args:
            **kwargs: The parameters to set.

        Returns:
            dict: A dictionary containing the parameters that have been changed.
        """
        changed_parameters = Calculator.set(self, **kwargs)

        if "run_dir" in self.parameters:
            self.run_dir = self.parameters["run_dir"]
            self.logger.info("run directory: %s is set from parameters", self.run_dir)
        if "relax_steps" in self.parameters:
            self.relax_steps = self.parameters["relax_steps"]
            self.logger.info("relaxation steps: %s is set from parameters", self.relax_steps)
        if "kim_potential" in self.parameters:
            self.kim_potential = self.parameters["kim_potential"]
            self.logger.info("kim potential: %s is set from parameters", self.kim_potential)

        return changed_parameters

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties=implemented_properties,
        system_changes=all_changes,
    ) -> None:
        """Calculate the properties of the system including static and relaxed energies.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the properties for.
            properties (tuple): The properties to calculate.
            system_changes (list): The system changes to calculate.
        """
        if atoms is None:
            atoms = self.atoms

        Calculator.calculate(self, atoms, properties, system_changes)

        if "energy" in properties:
            unrelaxed_results = self.run_lammps_energy(atoms, run_dir=self.run_dir)
            self.results["energy"] = unrelaxed_results[1]
            self.results["per_atom_energies"] = unrelaxed_results[2]

        if "relaxed_energy" in properties:
            relaxed_results = self.run_lammps_opt(atoms, run_dir=self.run_dir)
            self.results["relaxed_energy"] = relaxed_results[1]
            self.results["per_atom_energies"] = relaxed_results[2]


class LAMMPSSurfCalc(LAMMMPSCalc):
    """Custom LAMMPSSurfCalc class to calculate surface energy."""

    implemented_properties = (*LAMMMPSCalc.implemented_properties, "surface_energy")

    def __init__(self, *args, **kwargs):
        """Initialize the LAMMPSSurfCalc class."""
        super().__init__(*args, **kwargs)
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

    def get_surface_energy(self, atoms: ase.Atoms = None) -> float:
        """Get the surface energy of the system. Currently the same as the potential energy.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the surface energy for.

        Returns:
            float: The surface energy of the system.
        """
        if atoms is None:
            atoms = self.atoms

        return self.get_potential_energy(atoms=atoms)

    def set(self, **kwargs) -> dict:
        """Set parameters in key-value pairs. A dictionary containing the parameters that have been
        changed is returned. The special keyword 'parameters' can be used to read parameters from a
        file.

        Args:
            **kwargs: The parameters to set.

        Returns:
            dict: A dictionary containing the parameters that have been changed.
        """
        return LAMMMPSCalc.set(self, **kwargs)

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties=implemented_properties,
        system_changes=all_changes,
    ) -> None:
        """Caculate based on LAMMMPSCalc before add in surface energy calcs to results.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the properties for.
            properties (tuple): The properties to calculate.
            system_changes (list): The system changes to calculate.
        """
        if atoms is None:
            atoms = self.atoms

        LAMMMPSCalc.calculate(self, atoms, properties, system_changes)

        if "surface_energy" in properties:
            self.results["surface_energy"] = self.get_surface_energy(atoms=atoms)


class LAMMPSRunSurfCalc(LAMMPSRun):
    """LAMMPSRunSurfCalc class based on ASE LAMMPSRun to calculate surface energy."""

    implemented_properties = (*LAMMPSRun.implemented_properties, "surface_energy")

    def __init__(self, *args, **kwargs):
        """Initialize the LAMMPSRunSurfCalc class."""
        super().__init__(*args, **kwargs)
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

    def get_surface_energy(self, atoms: ase.Atoms = None) -> float:
        """Get the surface energy of the system. Currently the same as the potential energy.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the surface energy for.

        Returns:
            float: The surface energy of the system.
        """
        if atoms is None:
            atoms = self.atoms

        return self.get_potential_energy(atoms=atoms)

    def set(self, **kwargs) -> dict:
        """Set parameters in key-value pairs. A dictionary containing the parameters that have been
        changed is returned. The special keyword 'parameters' can be used to read parameters from a
        file.

        Args:
            **kwargs: The parameters to set.

        Returns:
            dict: A dictionary containing the parameters that have been changed.
        """
        return LAMMPSRun.set(self, **kwargs)

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties=implemented_properties,
        system_changes=all_changes,
    ) -> None:
        """Caculate based on LAMMPSRun before add in surface energy calcs to results.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the properties for.
            properties (tuple): The properties to calculate.
            system_changes (list): The system changes to calculate.
        """
        if atoms is None:
            atoms = self.atoms

        LAMMPSRun.calculate(self, atoms, properties, system_changes)

        if "surface_energy" in properties:
            self.results["surface_energy"] = self.get_surface_energy(atoms=atoms)
