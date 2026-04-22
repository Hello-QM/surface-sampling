"""Tests for the 3-layer surface Pourbaix correction module."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

# Load adsorbate_gibbs.py directly to avoid the mcmc package's catkit dependency.
_MODULE_PATH = Path(__file__).resolve().parents[1] / "mcmc" / "corrections" / "adsorbate_gibbs.py"
_spec = importlib.util.spec_from_file_location("_ads_gibbs_module", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ADS_G_INTRINSIC = _mod.ADS_G_INTRINSIC
DEFAULT_DELTA_O = _mod.DEFAULT_DELTA_O
DEFAULT_EPS_HBOND = _mod.DEFAULT_EPS_HBOND
count_adsorbates = _mod.count_adsorbates
count_hbonds = _mod.count_hbonds
slab_correction = _mod.slab_correction


# ---------------------------------------------------------------------------
# Tiny test fixtures
# ---------------------------------------------------------------------------

def _two_oh_isolated() -> Atoms:
    """Two *OH on top of isolated Ir atoms, 8 Å apart (no H-bonds possible)."""
    return Atoms(
        symbols=["Ir", "O", "H", "Ir", "O", "H"],
        positions=[
            [0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0],
            [8.0, 0.0, 0.0], [8.0, 0.0, 2.0], [8.0, 0.0, 3.0],
        ],
        cell=[20, 20, 20],
        pbc=True,
    )


def _two_oh_paired() -> Atoms:
    """Two *OH on top of Ir atoms arranged to form a single O-H...O bond."""
    return Atoms(
        symbols=["Ir", "O", "H", "Ir", "O", "H"],
        positions=[
            [0.0, 0.0, 0.0],   [0.0, 0.0, 2.0],   [0.9, 0.0, 2.4],
            [2.7, 0.0, 0.0],   [2.7, 0.0, 2.0],   [3.5, 0.0, 1.6],
        ],
        cell=[20, 20, 20],
        pbc=True,
    )


def _star_O() -> Atoms:
    """One *O (bare adsorbate) on top of an Ir atom."""
    return Atoms(
        symbols=["Ir", "O"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        cell=[20, 20, 20],
        pbc=True,
    )


def _star_OOH() -> Atoms:
    """One *OOH — inner O on Ir, terminal O bonded only to inner O and H."""
    return Atoms(
        symbols=["Ir", "O", "O", "H"],
        positions=[
            [0.0, 0.0, 0.0],    # Ir
            [0.0, 0.0, 2.0],    # inner O  (1 Ir neighbor, 1 O neighbor at 1.45 Å)
            [0.9, 0.0, 3.1],    # terminal O (0 Ir, 1 O neighbor, 1 H)
            [1.8, 0.0, 3.1],    # H
        ],
        cell=[20, 20, 20],
        pbc=True,
    )


def _star_H2O() -> Atoms:
    """One *H2O on top of an Ir atom."""
    return Atoms(
        symbols=["Ir", "O", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.2],
            [0.76, 0.0, 2.8],
            [-0.76, 0.0, 2.8],
        ],
        cell=[20, 20, 20],
        pbc=True,
    )


def _bulk_O_only() -> Atoms:
    """Bulk-like 3-coordinated O (coord_Ir == 3) — should NOT be counted as adsorbate."""
    return Atoms(
        symbols=["O", "Ir", "Ir", "Ir"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.9, 0.0, 0.0],
            [-1.9, 0.0, 0.0],
            [0.0, 0.0, 1.9],
        ],
        cell=[20, 20, 20],
        pbc=True,
    )


# ---------------------------------------------------------------------------
# Layer B: adsorbate identification
# ---------------------------------------------------------------------------

def test_count_adsorbates_isolated_two_OH():
    counts = count_adsorbates(_two_oh_isolated())
    assert counts == Counter({"OH": 2})


def test_count_adsorbates_single_O():
    counts = count_adsorbates(_star_O())
    assert counts == Counter({"O": 1})


def test_count_adsorbates_OOH():
    counts = count_adsorbates(_star_OOH())
    assert counts == Counter({"OOH": 1})


def test_count_adsorbates_H2O():
    counts = count_adsorbates(_star_H2O())
    assert counts == Counter({"H2O": 1})


def test_count_adsorbates_bulk_O_excluded():
    """An O with 3 Ir neighbors is bulk-like and must not be counted as adsorbate."""
    counts = count_adsorbates(_bulk_O_only())
    assert counts == Counter()


# ---------------------------------------------------------------------------
# Layer C: H-bond counting
# ---------------------------------------------------------------------------

def test_count_hbonds_isolated_zero():
    assert count_hbonds(_two_oh_isolated()) == 0


def test_count_hbonds_paired_one():
    assert count_hbonds(_two_oh_paired()) == 1


# ---------------------------------------------------------------------------
# Top-level: composed correction
# ---------------------------------------------------------------------------

def test_slab_correction_breakdown_isolated():
    atoms = _two_oh_isolated()
    total, bd = slab_correction(atoms, return_breakdown=True)

    expected_A = 2 * DEFAULT_DELTA_O                       # 2 O atoms total
    expected_B = 2 * ADS_G_INTRINSIC["OH"]                 # 2 *OH adsorbates
    expected_C = 0.0                                       # no H-bonds

    assert bd["n_O_total"] == 2
    assert bd["ads_counts"] == {"OH": 2}
    assert bd["n_hbonds"] == 0
    np.testing.assert_allclose(bd["A_oxide"], expected_A)
    np.testing.assert_allclose(bd["B_adsorbate"], expected_B)
    np.testing.assert_allclose(bd["C_hbond"], expected_C)
    np.testing.assert_allclose(total, expected_A + expected_B + expected_C)


def test_slab_correction_paired_has_hbond():
    """Paired-OH geometry produces exactly 1 H-bond × DEFAULT_EPS_HBOND."""
    total, bd = slab_correction(_two_oh_paired(), return_breakdown=True)

    assert bd["n_hbonds"] == 1
    np.testing.assert_allclose(bd["C_hbond"], DEFAULT_EPS_HBOND)


def test_isolated_vs_paired_differ_by_one_hbond():
    """Same composition, different geometry → correction differs by exactly ε_HB."""
    g_iso = slab_correction(_two_oh_isolated())
    g_pair = slab_correction(_two_oh_paired())
    np.testing.assert_allclose(g_pair - g_iso, DEFAULT_EPS_HBOND, atol=1e-10)


def test_override_delta_O_passes_through():
    """Custom Δ_O scales linearly with O count."""
    atoms = _two_oh_isolated()
    total, bd = slab_correction(atoms, delta_O=-0.5, return_breakdown=True)
    np.testing.assert_allclose(bd["A_oxide"], 2 * -0.5)
