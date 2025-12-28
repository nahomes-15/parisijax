#!/usr/bin/env python3
"""GPU-accelerated parallel tempering to properly equilibrate spin glass phase."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from parisijax.core import hamiltonian, mcmc, solver
from parisijax.analysis import overlap

# Try GPU, fall back to CPU
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

backend = jax.default_backend()
if 'gpu' in backend or 'cuda' in backend:
    print(f"✓ Running on GPU: {jax.devices()}")
    jax.config.update("jax_enable_x64", False)  # float32 for speed
else:
    print(f"Running on CPU: {jax.devices()}")
    jax.config.update("jax_enable_x64", True)  # Keep precision

print(f"Backend: {backend}\n")

N_SPINS = 500
SEED = 42


def banner(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def run_parallel_tempering_sampling(key, J, beta_target, n_temps=16, n_steps=50000):
    """Use parallel tempering to sample at target beta with proper equilibration.

    Strategy: Run PT across temperature ladder, extract samples from beta_target.
    """
    from parisijax.core.hamiltonian import random_spins

    # Create temperature ladder: geometric spacing from high T to low T
    beta_min = 0.2
    beta_max = beta_target
    betas = jnp.geomspace(beta_min, beta_max, n_temps)

    print(f"  Temperature ladder: {n_temps} replicas")
    print(f"  β range: [{beta_min:.2f}, {beta_max:.2f}]")
    print(f"  Target β = {beta_target:.2f} (index {n_temps-1})")

    # Initialize spins at all temperatures
    key, subkey = jax.random.split(key)
    spins_all_temps = random_spins(subkey, N_SPINS, n_samples=n_temps)

    # Run parallel tempering
    print(f"  Running {n_steps} PT steps...")

    t0 = time.time()

    # Collect samples from target temperature every 50 steps
    samples_collected = []

    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        spins_all_temps = mcmc.parallel_tempering_step(subkey, spins_all_temps, J, betas)

        # Collect sample from coldest temperature
        if step % 50 == 0 and step > n_steps // 2:  # Only after burnin
            samples_collected.append(spins_all_temps[-1])

        if step % 5000 == 0:
            print(f"    Step {step}/{n_steps}")

    jax.block_until_ready(spins_all_temps)
    t_pt = time.time() - t0

    print(f"  PT time: {t_pt:.2f} s")
    print(f"  Collected {len(samples_collected)} samples")

    return jnp.array(samples_collected), t_pt


def main():
    key = jax.random.PRNGKey(SEED)

    banner(f"GPU Parallel Tempering: N = {N_SPINS}")

    # Generate instance
    key, subkey = jax.random.split(key)
    J = hamiltonian.sample_couplings(subkey, N_SPINS, n_samples=1)[0]

    print(f"  System size: N = {N_SPINS}")

    beta_c = solver.find_critical_temperature()
    print(f"  Critical point: βc = {beta_c:.3f}")

    # ========================================================================
    # High Temperature (simple MCMC is fine)
    # ========================================================================
    banner("High T: β = 0.5 (Standard MCMC)")

    beta_high = 0.5

    key, subkey = jax.random.split(key)
    t0 = time.time()
    overlaps_high = overlap.sample_overlap_distribution(
        subkey, J, beta_high, n_samples=200, n_steps=5000, burnin=1000
    )
    jax.block_until_ready(overlaps_high)
    t_high = time.time() - t0

    qea_high = overlap.compute_edwards_anderson_parameter(overlaps_high)
    print(f"  qₑₐ = {qea_high:.4f}")
    print(f"  Time: {t_high:.2f} s")

    # ========================================================================
    # Low Temperature (use parallel tempering!)
    # ========================================================================
    banner("Low T: β = 2.0 (Parallel Tempering)")

    beta_low = 2.0

    # Run PT twice to get two independent samples for overlap
    key, subkey1 = jax.random.split(key)
    samples1, t1 = run_parallel_tempering_sampling(subkey1, J, beta_low, n_temps=16, n_steps=50000)

    key, subkey2 = jax.random.split(key)
    samples2, t2 = run_parallel_tempering_sampling(subkey2, J, beta_low, n_temps=16, n_steps=50000)

    # Compute overlaps between pairs
    n_pairs = min(len(samples1), len(samples2))
    overlaps_low = jax.vmap(overlap.compute_overlap)(samples1[:n_pairs], samples2[:n_pairs])

    qea_low = overlap.compute_edwards_anderson_parameter(overlaps_low)
    print(f"\n  qₑₐ = {qea_low:.4f}")
    print(f"  Total time: {t1 + t2:.2f} s")
    print(f"  Overlap range: [{jnp.min(overlaps_low):.3f}, {jnp.max(overlaps_low):.3f}]")

    # Check for transition
    if qea_low > 0.2:
        print(f"\n  ✓✓✓ STRONG SIGNAL: qₑₐ = {qea_high:.3f} → {qea_low:.3f}")
    elif qea_low > 0.05:
        print(f"\n  ✓ TRANSITION VISIBLE: qₑₐ increased to {qea_low:.3f}")
    else:
        print(f"\n  ⚠️  Still weak: {qea_low:.3f}. Need more PT steps.")

    # ========================================================================
    # Visualization
    # ========================================================================
    banner("Results")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # High T
    ax = axes[0]
    ax.hist(np.array(overlaps_high), bins=50, range=(-0.5, 0.5), density=True,
            alpha=0.7, edgecolor='black', linewidth=1.2, color='blue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel('Overlap q', fontsize=13)
    ax.set_ylabel('P(q)', fontsize=13)
    ax.set_title(f'High T: β = {beta_high}\nqₑₐ = {qea_high:.4f} (Paramagnetic)',
                 fontsize=13, weight='bold')
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, 'Standard MCMC', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Low T
    ax = axes[1]
    counts, bins, _ = ax.hist(np.array(overlaps_low), bins=50, range=(-0.5, 0.5), density=True,
                               alpha=0.7, edgecolor='black', linewidth=1.2, color='orange')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)

    # Highlight if we see a peak away from zero
    if qea_low > 0.1:
        peak_idx = np.argmax(counts)
        peak_q = (bins[peak_idx] + bins[peak_idx+1]) / 2
        if abs(peak_q) > 0.1:
            ax.axvline(peak_q, color='green', linestyle=':', linewidth=3, alpha=0.8)
            ax.text(peak_q, max(counts) * 0.9, f'q* ≈ {peak_q:.2f}',
                   ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.set_xlabel('Overlap q', fontsize=13)
    ax.set_ylabel('P(q)', fontsize=13)
    ax.set_title(f'Low T: β = {beta_low}\nqₑₐ = {qea_low:.4f} (Spin Glass)',
                 fontsize=13, weight='bold')
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, 'Parallel Tempering\n50k steps', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'SK Spin Glass Phase Transition (N = {N_SPINS}, GPU)',
                 fontsize=15, weight='bold')
    plt.tight_layout()

    output_path = Path(__file__).parent / "sk_gpu_pt.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    banner("Summary")
    print(f"  Device: {jax.devices()[0]}")
    print(f"  N = {N_SPINS}")
    print(f"  High T: Standard MCMC, qₑₐ = {qea_high:.4f}")
    print(f"  Low T: Parallel tempering (50k steps), qₑₐ = {qea_low:.4f}")
    print(f"  Phase transition strength: {qea_low / max(qea_high, 1e-6):.1f}× increase")

    if qea_low > 0.15:
        print(f"\n  🎉 SUCCESS: Clear spin glass phase detected!")
    elif qea_low > 0.05:
        print(f"\n  ✓ Transition visible (try 100k PT steps for stronger signal)")
    else:
        print(f"\n  Need longer PT runs or GPU isn't being used")


if __name__ == "__main__":
    main()
