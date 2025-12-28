#!/usr/bin/env python3
"""Fixed demo: Proper equilibration for spin glass phase transition."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from parisijax.core import hamiltonian, mcmc, solver
from parisijax.analysis import overlap

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Fixed parameters
N_SPINS = 200
SEED = 42


def banner(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def main():
    key = jax.random.PRNGKey(SEED)

    banner("Setup: Generate Single SK Instance")

    key, subkey = jax.random.split(key)
    J = hamiltonian.sample_couplings(subkey, N_SPINS, n_samples=1)[0]

    print(f"  N = {N_SPINS}")
    print(f"  Coupling check: max|J - Jᵀ| = {jnp.max(jnp.abs(J - J.T)):.2e}")

    # Critical temperature
    beta_c = solver.find_critical_temperature()
    print(f"  Critical point: βc = {beta_c:.3f} (Tc = {1/beta_c:.3f})")

    banner("Diagnostic: Self-Overlap Check")

    # Run one chain and check self-overlap
    key, subkey = jax.random.split(key)
    test_spins, _ = mcmc.run_mcmc(subkey, J, beta=1.0, n_steps=100, n_samples=1)
    self_overlap = overlap.compute_overlap(test_spins[0], test_spins[0])
    print(f"  Self-overlap q(σ,σ) = {self_overlap:.6f} (should be exactly 1.0)")

    if not jnp.isclose(self_overlap, 1.0):
        print("  ⚠️  WARNING: Self-overlap is not 1.0! Normalization bug detected.")
        return

    banner("High Temperature: β = 0.5 (Paramagnetic Phase)")

    beta_high = 0.5
    n_samples_high = 250
    n_steps_high = 2000
    burnin_high = 500

    print(f"  Temperature: T = {1/beta_high:.2f} (β = {beta_high})")
    print(f"  Running {2 * n_samples_high} chains × {n_steps_high + burnin_high} steps")

    key, subkey = jax.random.split(key)
    t0 = time.time()
    overlaps_high = overlap.sample_overlap_distribution(
        subkey, J, beta_high,
        n_samples=n_samples_high,
        n_steps=n_steps_high,
        burnin=burnin_high
    )
    jax.block_until_ready(overlaps_high)
    t_high = time.time() - t0

    qea_high = overlap.compute_edwards_anderson_parameter(overlaps_high)
    print(f"  qₑₐ = {qea_high:.4f} (expect ≈ 0 above Tc)")
    print(f"  Time: {t_high:.2f} s")
    print(f"  Overlap range: [{jnp.min(overlaps_high):.3f}, {jnp.max(overlaps_high):.3f}]")

    banner("Low Temperature: β = 2.0 (Spin Glass Phase)")

    beta_low = 2.0
    n_samples_low = 200
    n_steps_low = 20000  # 10× longer for spin glass!
    burnin_low = 5000    # More burnin

    print(f"  Temperature: T = {1/beta_low:.2f} (β = {beta_low})")
    print(f"  Running {2 * n_samples_low} chains × {n_steps_low + burnin_low} steps")
    print(f"  NOTE: Long runs needed for rugged energy landscape...")

    key, subkey = jax.random.split(key)
    t0 = time.time()
    overlaps_low = overlap.sample_overlap_distribution(
        subkey, J, beta_low,
        n_samples=n_samples_low,
        n_steps=n_steps_low,
        burnin=burnin_low
    )
    jax.block_until_ready(overlaps_low)
    t_low = time.time() - t0

    qea_low = overlap.compute_edwards_anderson_parameter(overlaps_low)
    print(f"  qₑₐ = {qea_low:.4f} (expect > 0 below Tc)")
    print(f"  Time: {t_low:.2f} s")
    print(f"  Overlap range: [{jnp.min(overlaps_low):.3f}, {jnp.max(overlaps_low):.3f}]")

    # Check if we see the transition
    if qea_low > 0.1:
        print(f"  ✓ Phase transition detected! qₑₐ increased from {qea_high:.3f} → {qea_low:.3f}")
    else:
        print(f"  ⚠️  Weak signal. May need longer equilibration or larger N.")

    banner("Visualization")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # High T
    ax = axes[0]
    ax.hist(np.array(overlaps_high), bins=50, range=(-1, 1), density=True,
            alpha=0.7, edgecolor='black', linewidth=1.2, color='blue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Overlap q', fontsize=13)
    ax.set_ylabel('P(q)', fontsize=13)
    ax.set_title(f'High T: β = {beta_high}, qₑₐ = {qea_high:.3f}\n(Paramagnetic)',
                 fontsize=13, weight='bold')
    ax.set_xlim(-1, 1)
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, f'N = {N_SPINS}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Low T
    ax = axes[1]
    ax.hist(np.array(overlaps_low), bins=50, range=(-1, 1), density=True,
            alpha=0.7, edgecolor='black', linewidth=1.2, color='orange')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Overlap q', fontsize=13)
    ax.set_ylabel('P(q)', fontsize=13)
    ax.set_title(f'Low T: β = {beta_low}, qₑₐ = {qea_low:.3f}\n(Spin Glass)',
                 fontsize=13, weight='bold')
    ax.set_xlim(-1, 1)
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, f'{n_steps_low} steps', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = Path(__file__).parent / "sk_phase_transition_fixed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    banner("Summary")
    print(f"  ✓ Same J used for both replica pairs (correct)")
    print(f"  ✓ Self-overlap verified: q(σ,σ) = 1.0")
    print(f"  ✓ High T equilibration: {n_steps_high} steps")
    print(f"  ✓ Low T equilibration: {n_steps_low} steps (10× longer)")
    print(f"  ✓ Phase transition: qₑₐ = {qea_high:.3f} → {qea_low:.3f}")

    if qea_low > 0.2:
        print(f"\n  🎉 SUCCESS: Clear spin glass signal!")
    elif qea_low > 0.05:
        print(f"\n  ⚠️  WEAK SIGNAL: Try N=500 or n_steps=50000")
    else:
        print(f"\n  ❌ NO SIGNAL: Need much longer runs or larger system")


if __name__ == "__main__":
    main()
