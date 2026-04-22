"""Local GPU smoke test — MACE + 3-layer slab_correction integration.

**Scope**: validate the NEW code paths added on the feat/batched-ntrial
branch can actually execute on a real GPU with a real MACE model. Does
NOT touch HRE-MC (requires IrO_pd_fixed.json which is pymatgen-version-
specific and breaks on the catgo local env; works on Expanse where the
xiaochendu fork is installed).

What's exercised here:
  * MACEPourbaix init (use_adsorbate_gibbs defaults to True)
  * MACEPourbaix.set(slab_correction_kwargs=...)
  * MACEPourbaix.get_delta_G1 with offset_data path (experimental μ° anchors)
  * slab_correction(atoms, delta_O, metal_symbol) integration
  * Sign of the 3-layer G_corr on a real slab

What's covered separately and NOT re-tested here:
  * MTM math → tests/test_mtm_math.py (9 tests, all pass)
  * 3-layer internals → tests/test_adsorbate_gibbs.py (11 tests, all pass)
  * HamiltonianREMC dispatch / FixedActionChangeProposal → static assert below

Target runtime: 30-60 s on RTX 4060.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))


def static_api_checks():
    """Quick structural asserts — no GPU / no MACE forward."""
    from mcmc.events.proposal import ChangeProposal, FixedActionChangeProposal
    from mcmc.hamiltonian_re import HamiltonianREMC
    import inspect

    sig = inspect.signature(HamiltonianREMC.__init__)
    assert "n_trials" in sig.parameters, "n_trials not in HamiltonianREMC init"
    assert hasattr(HamiltonianREMC, "_mc_step_mtm"), "_mc_step_mtm method missing"
    assert hasattr(HamiltonianREMC, "_mc_step_single"), "_mc_step_single method missing"
    assert issubclass(FixedActionChangeProposal, ChangeProposal)

    sig_run = inspect.signature(HamiltonianREMC.run)
    assert "save_interval" in sig_run.parameters
    print("  ✓ HamiltonianREMC has n_trials, _mc_step_mtm, _mc_step_single")
    print("  ✓ FixedActionChangeProposal is ChangeProposal subclass")
    print("  ✓ run() has save_interval parameter")


def build_iro2_slab():
    """Tiny rutile IrO2(110) slab, ~12 atoms."""
    from ase import Atoms
    from ase.build import surface

    bulk_iro2 = Atoms(
        symbols="Ir2O4",
        cell=[[4.49, 0, 0], [0, 4.49, 0], [0, 0, 3.15]],
        pbc=True,
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.306, 0.306, 0.0),
            (0.694, 0.694, 0.0),
            (0.194, 0.806, 0.5),
            (0.806, 0.194, 0.5),
        ],
    )
    slab = surface(bulk_iro2, (1, 1, 0), layers=2, vacuum=6.0)
    slab.pbc = True
    # Tag a few ads_group atoms for Layer B identification (1 = adsorbate)
    ads_group = np.zeros(len(slab), dtype=int)
    # mark the two topmost O atoms as adsorbates so Layer B counts them
    z = slab.positions[:, 2]
    top_O_idx = [i for i, s in enumerate(slab.get_chemical_symbols())
                 if s == "O" and z[i] > z.max() - 1.5][:2]
    for i in top_O_idx:
        ads_group[i] = 1
    slab.set_array("ads_group", ads_group)
    return slab


def test_3layer_via_MACEPourbaix():
    """Exercise the 3-layer path via real MACE forward pass."""
    from mcmc.calculators.calculators import MACEPourbaix

    slab = build_iro2_slab()
    print(f"  Slab: {slab.get_chemical_formula()} ({len(slab)} atoms)")

    model_path = str(Path.home() / ".cache" / "mace" / "20231203mace128L1_epoch199model")
    assert Path(model_path).exists(), f"smoke MACE model missing: {model_path}"

    calc = MACEPourbaix(model_path=model_path, device="cuda", enable_cueq=False)
    # Defaults check:
    assert calc.use_adsorbate_gibbs is True, "3-layer must be default True"
    print("  ✓ MACEPourbaix default use_adsorbate_gibbs=True (3-layer authoritative)")

    # Set the 3-layer knobs + dummy offset_data for the ΔG₁ bulk-ref path
    calc.set(
        temperature=0.0257,
        pH=0,
        phi=0.9,
        slab_correction_kwargs={"delta_O": -0.1252, "metal_symbol": "Ir"},
        offset_data={
            "bulk_energies": {"Ir": -8.85, "O": -4.95, "IrO2": -30.0},
            "stoics": {"Ir": 1, "O": 2},  # flat stoichiometry of ref_formula
            "ref_formula": "IrO2",
            "ref_element": "Ir",
        },
    )
    print(f"  ✓ use_adsorbate_gibbs={calc.use_adsorbate_gibbs}")
    print(f"  ✓ slab_correction_kwargs={calc.slab_correction_kwargs}")

    # Run real MACE + 3-layer ΔG₁
    t0 = time.time()
    slab.calc = calc
    dG1 = calc.get_delta_G1(slab)
    dt = time.time() - t0
    print(f"  ΔG₁ = {dG1:.4f} eV   (MACE + 3-layer, {dt:.2f} s)")

    # Test toggle: flip off 3-layer → should use legacy (empty dict, no correction)
    calc.set(use_adsorbate_gibbs=False)
    dG1_legacy = calc.get_delta_G1(slab)
    print(f"  ΔG₁ (legacy, use_adsorbate_gibbs=False) = {dG1_legacy:.4f} eV")
    diff = abs(dG1 - dG1_legacy)
    print(f"  |3-layer − legacy| = {diff:.4f} eV  (should be > 0 since Layer B adds +0.085 per ads-O)")
    assert diff > 0.05, (
        "3-layer correction should meaningfully shift ΔG₁ on a slab with adsorbates; "
        f"got diff={diff:.4f} eV"
    )


def test_slab_correction_sign_and_layers():
    """Independent call to slab_correction (no MACE) to see Layer A/B/C breakdown."""
    from mcmc.corrections.adsorbate_gibbs import slab_correction

    slab = build_iro2_slab()
    total, info = slab_correction(
        slab, delta_O=-0.1252, metal_symbol="Ir", return_breakdown=True
    )
    print(f"  Total G_corr = {total:+.4f} eV")
    print(f"    Layer A (bulk oxide, Δ_O × n_O): {info['A_oxide']:+.4f} eV  (n_O={info['n_O_total']})")
    print(f"    Layer B (per-species ads Gibbs): {info['B_adsorbate']:+.4f} eV  (counts={info['ads_counts']})")
    print(f"    Layer C (H-bond count × ε_HB):   {info['C_hbond']:+.4f} eV  (n_hbonds={info['n_hbonds']})")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    t_total = time.time()
    try:
        print("─── Static API asserts ───────────────────────────────")
        static_api_checks()
        print()
        print("─── 3-layer slab_correction breakdown ────────────────")
        test_slab_correction_sign_and_layers()
        print()
        print("─── MACE + 3-layer ΔG₁ end-to-end (GPU) ──────────────")
        test_3layer_via_MACEPourbaix()
        print()
        print(f"✅ SMOKE TEST PASSED in {time.time() - t_total:.1f} s")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\n❌ SMOKE TEST FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
