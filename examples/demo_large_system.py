#!/usr/bin/env python3
"""Large system demo: N=500 to see clear spin glass transition."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from parisijax.core import hamiltonian, mcmc, solver
from parisijax.analysis import overlap

jax.config.update("jax_enable_x64", True)

N_SPINS = 500  # Larger system
SEED = 42


def banner(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def main():
    key = jax.random.PRNGKey(SEED)

    banner(f"Large System Test: N = {N_SPINS}")

    key, subkey = jax.random.split(key)
    J = hamiltonian.sample_couplings(subkey, N_SPINS, n_samples=1)[0]

    print(f"  System size: N = {N_SPINS}")
    print(f"  Memory: J matrix = {J.nbytes / 1024**2:.1f} MB")

    beta_c = solver.find_critical_temperature()
    print(f"  βc = {beta_c:.3f}")

    banner("High T: β = 0.5")

    beta_high = 0.5
    n_samples = 100  # Fewer samples for large N
    n_steps = 3000

    key, subkey = jax.random.split(key)
    t0 = time.time()
    overlaps_high = overlap.sample_overlap_distribution(
        subkey, J, beta_high, n_samples=n_samples, n_steps=n_steps, burnin=1000
    )
    jax.block_until_ready(overlaps_high)
    t_high = time.time() - t0

    qea_high = overlap.compute_edwards_anderson_parameter(overlaps_high)
    print(f"  qₑₐ = {qea_high:.4f}")
    print(f"  Time: {t_high:.2f} s")

    banner("Low T: β = 2.0")

    beta_low = 2.0
    n_steps_low = 10000  # Still long but manageable

    key, subkey = jax.random.split(key)
    t0 = time.time()
    print(f"  Running {2 * n_samples} chains × {n_steps_low} steps...")
    overlaps_low = overlap.sample_overlap_distribution(
        subkey, J, beta_low, n_samples=n_samples, n_steps=n_steps_low, burnin=2000
    )
    jax.block_until_ready(overlaps_low)
    t_low = time.time() - t0

    qea_low = overlap.compute_edwards_anderson_parameter(overlaps_low)
    print(f"  qₑₐ = {qea_low:.4f}")
    print(f"  Time: {t_low:.2f} s")
    print(f"  Overlap range: [{jnp.min(overlaps_low):.3f}, {jnp.max(overlaps_low):.3f}]")

    if qea_low > 0.1:
        print(f"\n  ✓ TRANSITION DETECTED: qₑₐ increased {qea_high:.3f} → {qea_low:.3f}")
    else:
        print(f"\n  Still weak. N={N_SPINS} may need N=1000+ for clear signal.")

    banner("Visualization")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(np.array(overlaps_high), bins=50, range=(-0.5, 0.5), density=True,
            alpha=0.7, edgecolor='black', color='blue')
    ax.set_xlabel('Overlap q', fontsize=13)
    ax.set_ylabel('P(q)', fontsize=13)
    ax.set_title(f'β = {beta_high}: qₑₐ = {qea_high:.3f}', fontsize=13, weight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(np.array(overlaps_low), bins=50, range=(-0.5, 0.5), density=True,
            alpha=0.7, edgecolor='black', color='orange')
    ax.set_xlabel('Overlap q', fontsize=13)
    ax.set_ylabel('P(q)', fontsize=13)
    ax.set_title(f'β = {beta_low}: qₑₐ = {qea_low:.3f}', fontsize=13, weight='bold')
    ax.grid(alpha=0.3)

    plt.suptitle(f'SK Model: N = {N_SPINS}', fontsize=15, weight='bold')
    plt.tight_layout()

    output_path = Path(__file__).parent / f"sk_N{N_SPINS}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
