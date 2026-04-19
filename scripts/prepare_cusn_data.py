"""Prepare Cu-Sn-O thermodynamic data and pristine surface slab for Pourbaix sampling.

Fetches computed entries from the Materials Project, builds phase and Pourbaix
diagrams, computes bulk reference (offset) data, and cuts a surface slab from
the selected bulk structure.

Usage:
    python prepare_cusn_data.py --mp_api_key YOUR_KEY --save_folder data/CuSn_001/
    python prepare_cusn_data.py --elements Cu Sn O --bulk_mp_id mp-5765
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from ase.io import write as ase_write
from monty.serialization import dumpfn, loadfn
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixEntry
from pymatgen.core import Composition, Element, Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.ase import AseAtomsAdaptor


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Prepare Cu-Sn-O thermodynamic data for Pourbaix surface sampling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mp_api_key",
        type=str,
        default=None,
        help="Materials Project API key (or set MP_API_KEY env var)",
    )
    p.add_argument(
        "--elements",
        nargs="+",
        default=["Cu", "Sn", "O"],
        help="Elements in the system",
    )
    p.add_argument(
        "--bulk_mp_id",
        type=str,
        default=None,
        help="MP ID for bulk structure (auto-selected if not provided)",
    )
    p.add_argument(
        "--hkl",
        nargs=3,
        type=int,
        default=[0, 0, 1],
        help="Miller index for surface cut",
    )
    p.add_argument("--layers", type=int, default=5, help="Number of slab layers")
    p.add_argument("--fixed", type=int, default=3, help="Number of fixed bottom layers")
    p.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=[2, 2],
        help="Supercell size (a, b)",
    )
    p.add_argument("--vacuum", type=float, default=10.0, help="Vacuum in Angstroms")
    p.add_argument(
        "--save_folder",
        type=str,
        default="./data/CuSn_001/",
        help="Output directory",
    )
    return p.parse_args()


def fetch_entries(mpr, elements):
    """Fetch computed entries for the given element system from Materials Project.

    Args:
        mpr: MPRester instance.
        elements: List of element symbols.

    Returns:
        list[ComputedEntry]: GGA/GGA+U computed entries processed with MP2020 compatibility.
    """
    print(f"Fetching computed entries for {'-'.join(elements)} system ...")
    entries = mpr.get_entries_in_chemsys(
        elements,
        additional_criteria={"thermo_types": ["GGA_GGA+U"]},
    )
    compat = MaterialsProject2020Compatibility()
    entries = compat.process_entries(entries)
    print(f"  Retrieved {len(entries)} entries after compatibility processing")
    return entries


def build_phase_diagram(entries, elements):
    """Build a pymatgen PhaseDiagram from computed entries.

    Args:
        entries: List of ComputedEntry objects.
        elements: List of element symbols.

    Returns:
        PhaseDiagram
    """
    print("Building phase diagram ...")
    pd = PhaseDiagram(entries)
    stable_formulas = [e.composition.reduced_formula for e in pd.stable_entries]
    print(f"  Stable phases: {stable_formulas}")
    return pd


def build_pourbaix_diagram(mpr, elements):
    """Build a pymatgen PourbaixDiagram including solid and ion entries.

    Args:
        mpr: MPRester instance.
        elements: List of element symbols (must include O for aqueous chemistry).

    Returns:
        PourbaixDiagram
    """
    # Filter to non-O elements for Pourbaix query (O/H handled internally)
    non_oh_elements = [e for e in elements if e not in ("O", "H")]
    print(f"Building Pourbaix diagram for {non_oh_elements} ...")
    pbx_entries = mpr.get_pourbaix_entries(non_oh_elements)
    print(f"  Retrieved {len(pbx_entries)} Pourbaix entries")
    pbx = PourbaixDiagram(pbx_entries, comp_dict={e: 1.0 / len(non_oh_elements) for e in non_oh_elements})
    return pbx


def compute_offset_data(pd, bulk_formula, ref_element):
    """Compute bulk reference energies and stoichiometric offset data.

    The offset data allows converting slab total energies to surface formation
    energies by subtracting the appropriate bulk reference.

    Args:
        pd: PhaseDiagram.
        bulk_formula: Formula of the reference bulk phase (e.g., "CuSnO3").
        ref_element: Reference element for the stoichiometric subtraction.

    Returns:
        dict: offset_data dictionary.
    """
    print(f"Computing offset data for {bulk_formula}, ref_element={ref_element} ...")
    comp = Composition(bulk_formula)
    elements_in_formula = [str(el) for el in comp.elements]

    # Bulk energies: per-formula-unit energy for each elemental endpoint and the compound
    bulk_energies = {}
    for el in elements_in_formula:
        if el == "O":
            # O reference: energy per atom of the O2 stable entry
            o_comp = Composition("O")
            bulk_energies["O"] = pd.get_reference_energy_per_atom(o_comp)
        else:
            el_comp = Composition(el)
            bulk_energies[el] = pd.get_reference_energy_per_atom(el_comp)

    # Compound energy per formula unit
    bulk_entry = None
    for entry in pd.stable_entries:
        if entry.composition.reduced_formula == comp.reduced_formula:
            bulk_entry = entry
            break
    if bulk_entry is None:
        # Fall back: find the entry closest to the hull for the target composition
        print(f"  WARNING: {bulk_formula} not found among stable entries, searching all entries ...")
        for entry in pd.all_entries:
            if entry.composition.reduced_formula == comp.reduced_formula:
                bulk_entry = entry
                break
    if bulk_entry is None:
        raise ValueError(
            f"No entry found for {bulk_formula} in the phase diagram. "
            "Provide a valid bulk_mp_id or check your elements."
        )

    # Energy per formula unit
    reduced_comp = bulk_entry.composition.reduced_composition
    n_fu = bulk_entry.composition.num_atoms / reduced_comp.num_atoms
    bulk_energy_per_fu = bulk_entry.energy / n_fu
    bulk_energies[comp.reduced_formula] = bulk_energy_per_fu

    # Stoichiometry dict
    stoics = {str(el): int(comp[el]) for el in comp.elements}

    # Stoichiometric correction dict (for chemical potential deviations)
    # stoidict[el] = correction per excess atom of el relative to ref_element
    stoidict = {}
    ref_count = stoics[ref_element]
    for el in stoics:
        if el != ref_element:
            stoidict[el] = bulk_energies.get(el, 0.0) - stoics[el] / ref_count * bulk_energies.get(
                ref_element, 0.0
            )
        else:
            stoidict[el] = 0.0
    # Offset: bulk energy per ref_element atom
    stoidict["offset"] = bulk_energy_per_fu / ref_count

    offset_data = {
        "bulk_energies": bulk_energies,
        "stoidict": stoidict,
        "stoics": stoics,
        "ref_formula": comp.reduced_formula,
        "ref_element": ref_element,
    }
    print(f"  bulk_energies: {bulk_energies}")
    print(f"  stoics: {stoics}")
    return offset_data


def get_bulk_structure(mpr, elements, bulk_mp_id=None):
    """Get the bulk structure from Materials Project.

    If bulk_mp_id is provided, fetch that specific structure. Otherwise,
    find the most stable ternary phase in the element system.

    Args:
        mpr: MPRester instance.
        elements: List of element symbols.
        bulk_mp_id: Optional MP material ID.

    Returns:
        tuple: (Structure, mp_id, formula)
    """
    if bulk_mp_id:
        print(f"Fetching bulk structure {bulk_mp_id} ...")
        doc = mpr.materials.summary.get_data_by_id(bulk_mp_id)
        return doc.structure, bulk_mp_id, doc.formula_pretty

    # Auto-select: find the most stable ternary compound
    print("Auto-selecting most stable ternary compound ...")
    non_oh_elements = [e for e in elements if e != "O"]
    docs = mpr.materials.summary.search(
        elements=elements,
        num_elements=len(elements),
        fields=["material_id", "formula_pretty", "energy_above_hull", "structure"],
    )
    if not docs:
        raise ValueError(f"No ternary compounds found for {elements}")

    # Sort by energy above hull
    docs_sorted = sorted(docs, key=lambda d: d.energy_above_hull)
    best = docs_sorted[0]
    print(f"  Selected: {best.formula_pretty} ({best.material_id}), "
          f"E_above_hull = {best.energy_above_hull:.4f} eV/atom")
    return best.structure, str(best.material_id), best.formula_pretty


def cut_slab(bulk_structure, hkl, layers, fixed, size, vacuum):
    """Cut a surface slab from the bulk structure using catkit.

    Args:
        bulk_structure: pymatgen Structure.
        hkl: Miller indices [h, k, l].
        layers: Number of slab layers.
        fixed: Number of fixed layers.
        size: Supercell size (a, b).
        vacuum: Vacuum thickness in Angstroms.

    Returns:
        ase.Atoms: Pristine slab.
    """
    from mcmc.utils.slab import surface_from_bulk

    adaptor = AseAtomsAdaptor()
    bulk_atoms = adaptor.get_atoms(bulk_structure)
    print(f"Cutting slab: hkl={hkl}, layers={layers}, fixed={fixed}, size={size}, vacuum={vacuum} ...")
    slab, surface_atoms = surface_from_bulk(
        bulk_atoms,
        miller_index=hkl,
        layers=layers,
        fixed=fixed,
        size=tuple(size),
        vacuum=vacuum,
    )
    n_surface = sum(surface_atoms)
    print(f"  Slab has {len(slab)} atoms, {n_surface} surface atoms")
    print(f"  Cell: {slab.cell.cellpar()[:3]} Å, angles: {slab.cell.cellpar()[3:]}")
    return slab


def main():
    args = parse_args()

    # Resolve API key
    api_key = args.mp_api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        print("ERROR: Materials Project API key required. Set --mp_api_key or MP_API_KEY env var.")
        sys.exit(1)

    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    with MPRester(api_key) as mpr:
        # 1. Fetch entries and build phase diagram
        entries = fetch_entries(mpr, args.elements)
        pd = build_phase_diagram(entries, args.elements)

        # 2. Build Pourbaix diagram
        pbx = build_pourbaix_diagram(mpr, args.elements)

        # 3. Get bulk structure
        bulk_structure, mp_id, formula = get_bulk_structure(
            mpr, args.elements, args.bulk_mp_id
        )

    # 4. Compute offset data
    reduced_formula = Composition(formula).reduced_formula
    # Choose reference element: the rarest non-O element in the formula
    comp = Composition(reduced_formula)
    non_o_elements = [str(el) for el in comp.elements if str(el) != "O"]
    ref_element = min(non_o_elements, key=lambda el: comp[el])
    offset_data = compute_offset_data(pd, reduced_formula, ref_element)

    # 5. Cut surface slab
    slab = cut_slab(bulk_structure, args.hkl, args.layers, args.fixed, args.size, args.vacuum)

    # === Save all outputs ===
    print(f"\nSaving outputs to {save_folder}/ ...")

    # Phase diagram
    pd_path = save_folder / "CuSn_pd.json"
    dumpfn(pd, pd_path)
    print(f"  Phase diagram      → {pd_path}")

    # Pourbaix diagram
    pbx_path = save_folder / "CuSn_pbx.json"
    dumpfn(pbx, pbx_path)
    print(f"  Pourbaix diagram   → {pbx_path}")

    # Offset data
    offset_path = save_folder / "CuSn_offset_data.json"
    dumpfn(offset_data, offset_path)
    print(f"  Offset data        → {offset_path}")

    # Pristine slab (pickle for MCMC pipeline)
    hkl_str = "".join(str(i) for i in args.hkl)
    slab_pkl_path = save_folder / f"CuSn_{hkl_str}_pristine.pkl"
    with open(slab_pkl_path, "wb") as f:
        pickle.dump(slab, f)
    print(f"  Pristine slab .pkl → {slab_pkl_path}")

    # Pristine slab (CIF for visualization)
    slab_cif_path = save_folder / f"CuSn_{hkl_str}_pristine.cif"
    ase_write(str(slab_cif_path), slab)
    print(f"  Pristine slab .cif → {slab_cif_path}")

    # Bulk structure (CIF)
    bulk_cif_path = save_folder / "CuSn_bulk.cif"
    bulk_structure.to(filename=str(bulk_cif_path))
    print(f"  Bulk structure     → {bulk_cif_path}")

    print(f"\nDone! Selected bulk: {formula} ({mp_id})")
    print(f"Offset ref formula: {reduced_formula}, ref element: {ref_element}")


if __name__ == "__main__":
    main()
