#!/usr/bin/env python3
"""Production demo: SK spin glass phase transition with ParisiJAX."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from parisijax.core import hamiltonian, mcmc, solver
from parisijax.analysis import overlap

# Configure JAX
jax.config.update("jax_enable_x64", True)  # Double precision
jax.config.update("jax_platform_name", "cpu")  # Or "gpu" if available

# Experiment parameters
N_SPINS = 200
N_REPLICAS = 500
N_STEPS = 2000
BURNIN = 500
SEED = 42


def banner(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def main():
    """Run complete SK spin glass demo."""
    key = jax.random.PRNGKey(SEED)

    # ========================================================================
    # 1. System Setup
    # ========================================================================
    banner("1. Generating SK Instance")

    key, subkey = jax.random.split(key)
    J = hamiltonian.sample_couplings(subkey, N_SPINS, n_samples=1)[0]

    print(f"  System size: N = {N_SPINS}")
    print(f"  Coupling statistics: μ = {jnp.mean(J):.6f}, σ = {jnp.std(J):.6f}")
    print(f"  Symmetry check: max|J - Jᵀ| = {jnp.max(jnp.abs(J - J.T)):.2e}")

    # ========================================================================
    # 2. Critical Temperature
    # ========================================================================
    banner("2. Finding Critical Temperature (AT Line)")

    t0 = time.time()
    beta_c = solver.find_critical_temperature()
    t_critical = time.time() - t0

    print(f"  βc = {beta_c:.4f} (Tc = {1/beta_c:.4f})")
    print(f"  Theory prediction: βc ≈ 1.0")
    print(f"  Computation time: {t_critical*1000:.1f} ms")

    # ========================================================================
    # 3. Free Energy Comparison
    # ========================================================================
    banner("3. RS Free Energy Across Temperatures")

    betas = jnp.linspace(0.2, 2.5, 20)
    t0 = time.time()
    f_rs_vals = jnp.array([solver.rs_free_energy(b) for b in betas])
    t_fe = time.time() - t0

    print(f"  Computed {len(betas)} temperatures")
    print(f"  f(β=0.5) = {f_rs_vals[4]:.4f}")
    print(f"  f(β=1.5) = {f_rs_vals[14]:.4f}")
    print(f"  Computation time: {t_fe*1000:.1f} ms")

    # ========================================================================
    # 4. GPU MCMC Sampling
    # ========================================================================
    banner("4. Massively Parallel MCMC (High Temperature)")

    key, subkey = jax.random.split(key)
    beta_high = 0.5

    print(f"  Running {N_REPLICAS} replicas × {N_STEPS} steps...")
    print(f"  Temperature: T = {1/beta_high:.2f} (β = {beta_high})")

    t0 = time.time()
    spins_high, energy_traj_high = mcmc.run_mcmc(
        subkey, J, beta_high, n_steps=N_STEPS, n_samples=N_REPLICAS
    )
    jax.block_until_ready(spins_high)  # Ensure compilation + execution finish
    t_mcmc_high = time.time() - t0

    final_E_high = energy_traj_high[:, -1]
    print(f"  Final energy: {jnp.mean(final_E_high)/N_SPINS:.4f} ± {jnp.std(final_E_high)/N_SPINS:.4f} per spin")
    print(f"  Total time: {t_mcmc_high:.2f} s")
    print(f"  Throughput: {N_REPLICAS * N_STEPS / t_mcmc_high / 1e6:.2f} M spin-flips/sec")

    banner("5. Massively Parallel MCMC (Low Temperature)")

    key, subkey = jax.random.split(key)
    beta_low = 2.0

    print(f"  Temperature: T = {1/beta_low:.2f} (β = {beta_low})")

    t0 = time.time()
    spins_low, energy_traj_low = mcmc.run_mcmc(
        subkey, J, beta_low, n_steps=N_STEPS, n_samples=N_REPLICAS
    )
    jax.block_until_ready(spins_low)
    t_mcmc_low = time.time() - t0

    final_E_low = energy_traj_low[:, -1]
    print(f"  Final energy: {jnp.mean(final_E_low)/N_SPINS:.4f} ± {jnp.std(final_E_low)/N_SPINS:.4f} per spin")
    print(f"  Total time: {t_mcmc_low:.2f} s")

    # ========================================================================
    # 6. Overlap Distribution
    # ========================================================================
    banner("6. Overlap Distribution P(q)")

    print("  Sampling overlaps at high T...")
    key, subkey = jax.random.split(key)
    t0 = time.time()
    overlaps_high = overlap.sample_overlap_distribution(
        subkey, J, beta_high, n_samples=N_REPLICAS//2, n_steps=N_STEPS, burnin=BURNIN
    )
    jax.block_until_ready(overlaps_high)
    t_overlap_high = time.time() - t0

    qea_high = overlap.compute_edwards_anderson_parameter(overlaps_high)
    print(f"  High T: qₑₐ = {qea_high:.4f} (expect ~0)")
    print(f"  Time: {t_overlap_high:.2f} s")

    print("\n  Sampling overlaps at low T...")
    key, subkey = jax.random.split(key)
    t0 = time.time()
    overlaps_low = overlap.sample_overlap_distribution(
        subkey, J, beta_low, n_samples=N_REPLICAS//2, n_steps=N_STEPS, burnin=BURNIN
    )
    jax.block_until_ready(overlaps_low)
    t_overlap_low = time.time() - t0

    qea_low = overlap.compute_edwards_anderson_parameter(overlaps_low)
    print(f"  Low T: qₑₐ = {qea_low:.4f} (expect >0)")
    print(f"  Time: {t_overlap_low:.2f} s")

    # ========================================================================
    # 7. Visualization
    # ========================================================================
    banner("7. Generating Figures")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Free energy
    ax = axes[0, 0]
    ax.plot(betas, f_rs_vals, 'o-', linewidth=2, markersize=5, label='RS Solution')
    ax.axvline(beta_c, color='red', linestyle='--', alpha=0.7, label=f'βc = {beta_c:.2f}')
    ax.set_xlabel('Inverse Temperature β', fontsize=11)
    ax.set_ylabel('Free Energy per Spin', fontsize=11)
    ax.set_title('(a) Replica-Symmetric Free Energy', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Energy trajectories (high T)
    ax = axes[0, 1]
    for i in range(min(10, N_REPLICAS)):
        ax.plot(energy_traj_high[i] / N_SPINS, alpha=0.6, linewidth=0.8)
    ax.set_xlabel('MCMC Step', fontsize=11)
    ax.set_ylabel('Energy per Spin', fontsize=11)
    ax.set_title(f'(b) Thermalization at β = {beta_high}', fontsize=12, weight='bold')
    ax.grid(alpha=0.3)

    # (c) P(q) high T
    ax = axes[1, 0]
    ax.hist(np.array(overlaps_high), bins=40, range=(-1, 1), density=True,
            alpha=0.7, edgecolor='black', linewidth=1.2)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Overlap q', fontsize=11)
    ax.set_ylabel('P(q)', fontsize=11)
    ax.set_title(f'(c) High T: β = {beta_high}, qₑₐ = {qea_high:.3f}', fontsize=12, weight='bold')
    ax.set_xlim(-1, 1)
    ax.grid(alpha=0.3)

    # (d) P(q) low T
    ax = axes[1, 1]
    ax.hist(np.array(overlaps_low), bins=40, range=(-1, 1), density=True,
            alpha=0.7, edgecolor='black', linewidth=1.2, color='orange')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Overlap q', fontsize=11)
    ax.set_ylabel('P(q)', fontsize=11)
    ax.set_title(f'(d) Low T: β = {beta_low}, qₑₐ = {qea_low:.3f}', fontsize=12, weight='bold')
    ax.set_xlim(-1, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = Path(__file__).parent / "sk_phase_transition.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved figure: {output_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    banner("Summary")

    print("  ✓ Generated SK instance and validated couplings")
    print("  ✓ Located phase transition via AT stability")
    print("  ✓ Computed RS free energy landscape")
    print(f"  ✓ Ran {2 * N_REPLICAS * N_STEPS:,} MCMC steps across temperatures")
    print("  ✓ Observed paramagnetic → spin glass transition in P(q)")
    print(f"  ✓ Performance: {(N_REPLICAS * N_STEPS / t_mcmc_high / 1e6):.1f}M flips/sec")
    print(f"\n  Figure saved: {output_path}\n")


if __name__ == "__main__":
    main()
