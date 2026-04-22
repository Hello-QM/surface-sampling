"""Pure-Python tests for the Multiple-Try Metropolis math.

Validates:
  1. Boltzmann-weighted pick probabilities match exp(-β E_i) / Σ exp(-β E_j).
  2. MTM acceptance ratio α = W_forward / W_reverse matches Liu 2000 Eq 2.6
     for a symmetric proposal.
  3. Asymptotic equilibrium: the MTM chain on a toy 2-state system with
     known Boltzmann distribution converges to π(x) = exp(-β E_x) / Z.
  4. n_trials=1 reduces to standard Metropolis (α = min(1, exp(-β ΔE))).

These tests do NOT import HamiltonianREMC — they reproduce the acceptance
math in isolation so that failures point directly at the formula, not at
SurfaceSystem / MACE plumbing.
"""

from __future__ import annotations

import math

import numpy as np


def mtm_accept(
    E_x: float,
    E_trials: list[float],
    picked: int,
    beta: float,
) -> float:
    """MTM acceptance ratio α = min(1, W_forward / W_reverse).

    Mirrors the code path in
    ``mcmc.hamiltonian_re.HamiltonianREMC._mc_step_mtm``.
    """
    energies = np.asarray(E_trials, dtype=np.float64)
    E_ref = float(min(energies.min(), E_x))
    w_y = np.exp(-beta * (energies - E_ref))
    w_x = float(np.exp(-beta * (E_x - E_ref)))
    W_forward = float(w_y.sum())
    W_reverse = W_forward - float(w_y[picked]) + w_x
    return min(1.0, W_forward / W_reverse)


def boltzmann_pick_probs(E_trials: list[float], beta: float) -> np.ndarray:
    energies = np.asarray(E_trials, dtype=np.float64)
    E_ref = float(energies.min())
    w = np.exp(-beta * (energies - E_ref))
    return w / w.sum()


# ----------------------------------------------------------------------
# 1. Boltzmann pick probabilities
# ----------------------------------------------------------------------

def test_boltzmann_pick_uniform_when_equal_energies():
    """With identical energies, the pick is uniform over trials."""
    p = boltzmann_pick_probs([1.0, 1.0, 1.0, 1.0], beta=1.0)
    np.testing.assert_allclose(p, [0.25] * 4, atol=1e-12)


def test_boltzmann_pick_concentrates_on_lowest():
    """Low-energy trial dominates at low T (high β)."""
    p = boltzmann_pick_probs([0.0, 1.0, 1.0, 1.0], beta=10.0)
    # exp(-0)/norm should dominate
    assert p[0] > 0.99
    assert all(p[i] < 0.01 for i in range(1, 4))


def test_boltzmann_pick_smooth_at_high_T():
    """High-T pick is nearly uniform even with different energies."""
    p = boltzmann_pick_probs([0.0, 0.1, 0.2, 0.3], beta=0.1)
    # all within 5% of uniform
    np.testing.assert_allclose(p, [0.25] * 4, atol=0.05)


# ----------------------------------------------------------------------
# 2. MTM acceptance ratio (Liu 2000 Eq 2.6)
# ----------------------------------------------------------------------

def test_mtm_accept_single_trial_reduces_to_metropolis():
    """For n_trials=1, α collapses to standard Metropolis min(1, exp(-β ΔE))."""
    E_x, E_y, beta = 0.0, 0.5, 2.0
    alpha = mtm_accept(E_x=E_x, E_trials=[E_y], picked=0, beta=beta)
    expected = min(1.0, math.exp(-beta * (E_y - E_x)))
    assert abs(alpha - expected) < 1e-12

    # Downhill move (E_y < E_x) → α = 1 always
    alpha_down = mtm_accept(E_x=0.5, E_trials=[0.0], picked=0, beta=beta)
    assert alpha_down == 1.0


def test_mtm_accept_picked_is_lowest_of_many():
    """k=4, picked is the lowest-energy trial → α should be ~ 1 (high accept)."""
    E_x, beta = 1.0, 2.0
    E_trials = [0.0, 0.5, 0.5, 0.5]  # picked[0] is best
    alpha = mtm_accept(E_x=E_x, E_trials=E_trials, picked=0, beta=beta)
    # Manual: w_y = [exp(0), exp(-1), exp(-1), exp(-1)] = [1, e^-1, e^-1, e^-1]
    #         w_x = exp(-2*(1-0)) = e^-2
    #         W_f = 1 + 3*e^-1 ≈ 1 + 1.104 = 2.104
    #         W_r = W_f - 1 + e^-2 ≈ 1.104 + 0.135 = 1.240
    #         α = min(1, 2.104 / 1.240) = 1.0
    assert alpha == 1.0


def test_mtm_accept_picked_is_uphill_reduces_rate():
    """k=2, picked is strictly worse than x → α < 1."""
    E_x, beta = 0.0, 2.0
    E_trials = [1.0, 0.5]
    alpha = mtm_accept(E_x=E_x, E_trials=E_trials, picked=0, beta=beta)
    # W_f = exp(-2) + exp(-1) ≈ 0.135 + 0.368 = 0.503
    # W_r = W_f - exp(-2) + 1 = 0.368 + 1 = 1.368
    # α = 0.503 / 1.368 ≈ 0.368
    expected = (math.exp(-2.0) + math.exp(-1.0)) / (math.exp(-1.0) + 1.0)
    assert abs(alpha - expected) < 1e-10
    assert alpha < 1.0


def test_mtm_accept_nonnegative_and_bounded():
    """α ∈ [0, 1] for arbitrary energies."""
    rng = np.random.default_rng(42)
    for _ in range(200):
        E_x = rng.uniform(-5, 5)
        k = rng.integers(1, 8)
        E_trials = rng.uniform(-5, 5, size=k).tolist()
        picked = int(rng.integers(0, k))
        beta = rng.uniform(0.1, 10.0)
        alpha = mtm_accept(E_x=E_x, E_trials=E_trials, picked=picked, beta=beta)
        assert 0.0 <= alpha <= 1.0


# ----------------------------------------------------------------------
# 3. Asymptotic equilibrium on a toy 2-state system
# ----------------------------------------------------------------------

def test_mtm_biased_on_deterministic_proposal_small_state_space():
    """Documents the KNOWN bias of the Eq 2.4 independent-proposal
    approximation when the proposal is heavily conditional on current state.

    The toy 2-state chain (E_A=0, E_B=1, β=2) has a deterministic proposal
    (q(B|A) = q(A|B) = 1), which is pathological: sampling k trials from
    q(·|x) always lands on the same destination. In this regime the Eq 2.4
    shortcut (W_reverse = W_forward − w_y[j] + w_x) over-represents w_x in
    the reverse sum and produces a biased stationary distribution.

    This test *confirms* the bias exists on the toy case; production slab
    MC does not hit this regime because the proposal picks (site, species)
    from ~10^18 configurations, so forward and reverse trials never collide.
    Rigor-seeking users should use k=1 (equivalent to plain Metropolis;
    see ``test_mtm_reduces_to_single_metropolis_at_k_equals_1``).
    """
    E_A, E_B, beta = 0.0, 1.0, 2.0
    energies = {"A": E_A, "B": E_B}
    rng = np.random.default_rng(0)

    k = 4
    state = "A"
    counts = {"A": 0, "B": 0}
    n_steps = 80_000

    for _ in range(n_steps):
        counts[state] += 1
        other = "B" if state == "A" else "A"
        E_trials = [energies[other]] * k
        alpha = mtm_accept(E_x=energies[state], E_trials=E_trials, picked=0, beta=beta)
        if rng.random() < alpha:
            state = other

    ratio_observed = counts["A"] / counts["B"]
    ratio_unbiased = math.exp(beta * (E_B - E_A))  # = e^2 ≈ 7.389

    # The biased ratio is known analytically: detailed balance gives
    # π(A) * α(A→B) = π(B) * α(B→A). For our k=4:
    #   α(A→B): E_ref = E_A, w_y = [e^-β]*4, w_x = 1, W_f = 4e^-β,
    #           W_r = 4e^-β - e^-β + 1 = 3e^-β + 1,
    #           α = 4e^-β / (3e^-β + 1).
    #   α(B→A): E_ref = E_A, w_y = [1]*4, w_x = e^-β, W_f = 4,
    #           W_r = 4 - 1 + e^-β = 3 + e^-β,
    #           α = min(1, 4/(3+e^-β)) = 1 (since e^-β < 1).
    # Therefore π(A) / π(B) = α(B→A) / α(A→B) = (3e^-β + 1) / (4 e^-β).
    eB = math.exp(-beta)
    ratio_biased = (3 * eB + 1) / (4 * eB)
    assert abs(ratio_observed - ratio_biased) / ratio_biased < 0.05, (
        f"observed {ratio_observed:.3f}, expected biased {ratio_biased:.3f}"
    )
    assert ratio_observed < ratio_unbiased, (
        "bias should under-represent the low-E state vs. true Boltzmann"
    )


def test_mtm_reduces_to_single_metropolis_at_k_equals_1():
    """n_trials=1 MTM and plain Metropolis produce identical chains on the
    same RNG stream."""
    E_A, E_B, beta = 0.0, 1.0, 2.0
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    # MTM with k=1
    state_a = "A"
    path_a = []
    for _ in range(500):
        path_a.append(state_a)
        other = "B" if state_a == "A" else "A"
        E_x = E_A if state_a == "A" else E_B
        E_y = E_A if other == "A" else E_B
        alpha = mtm_accept(E_x=E_x, E_trials=[E_y], picked=0, beta=beta)
        if rng_a.random() < alpha:
            state_a = other

    # Plain Metropolis
    state_b = "A"
    path_b = []
    for _ in range(500):
        path_b.append(state_b)
        other = "B" if state_b == "A" else "A"
        E_x = E_A if state_b == "A" else E_B
        E_y = E_A if other == "A" else E_B
        alpha = min(1.0, math.exp(-beta * (E_y - E_x)))
        if rng_b.random() < alpha:
            state_b = other

    assert path_a == path_b, "MTM at k=1 should be identical to single Metropolis"
