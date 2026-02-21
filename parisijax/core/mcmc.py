"""Monte Carlo sampling: Metropolis, Gibbs, and parallel tempering.

Convention:
    - Metropolis ``n_steps`` counts **single-spin flips** (one random spin per step).
    - Gibbs ``n_steps`` counts **full sweeps** (all N spins updated sequentially).
    - ``_metropolis_sweep`` performs N single-spin flips in one call, making
      one Metropolis sweep comparable to one Gibbs sweep.
    - ``run_mcmc`` with method='metropolis' uses single-flip steps by default.
      For sweep-based Metropolis, use ``run_mcmc`` with method='metropolis_sweep'.
"""

from functools import partial
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap
from jax.typing import ArrayLike

# ============================================================================
# Single-Spin Update Rules
# ============================================================================

@jit
def _metropolis_step(key: ArrayLike, spins: jnp.ndarray, J: jnp.ndarray, beta: float, h: float) -> jnp.ndarray:
    """Single Metropolis update: flip random spin with acceptance min(1, exp(-beta dE))."""
    n_spins = spins.shape[0]

    # Select random spin
    key, subkey = random.split(key)
    i = random.randint(subkey, (), 0, n_spins)

    # Compute dE = 2*sigma_i * h_i where h_i = (sum_j J_ij sigma_j)/sqrt(N) + h
    local_field = jnp.dot(J[i], spins) / jnp.sqrt(n_spins) + h
    delta_E = 2.0 * spins[i] * local_field

    # Accept with Metropolis probability
    key, subkey = random.split(key)
    accept = random.uniform(subkey) < jnp.exp(-beta * delta_E)

    return spins.at[i].set(lax.select(accept, -spins[i], spins[i]))


@jit
def _metropolis_sweep(key: ArrayLike, spins: jnp.ndarray, J: jnp.ndarray, beta: float, h: float) -> jnp.ndarray:
    """Full Metropolis sweep: N single-spin flip attempts via lax.scan.

    One sweep proposes flipping each of N randomly chosen spins (with
    replacement), making it comparable to one Gibbs sweep in terms
    of computational work per call.
    """
    n_spins = spins.shape[0]

    def single_flip(carry, _):
        spins_cur, key_cur = carry
        key_cur, k1, k2 = random.split(key_cur, 3)

        i = random.randint(k1, (), 0, n_spins)
        local_field = jnp.dot(J[i], spins_cur) / jnp.sqrt(n_spins) + h
        delta_E = 2.0 * spins_cur[i] * local_field
        accept = random.uniform(k2) < jnp.exp(-beta * delta_E)
        spins_cur = spins_cur.at[i].set(
            lax.select(accept, -spins_cur[i], spins_cur[i])
        )
        return (spins_cur, key_cur), None

    (spins_out, _), _ = lax.scan(single_flip, (spins, key), None, length=n_spins)
    return spins_out


@jit
def _gibbs_step(key: ArrayLike, spins: jnp.ndarray, J: jnp.ndarray, beta: float, h: float) -> jnp.ndarray:
    """Sweep all spins with Gibbs sampling: sigma_i ~ exp(beta h_i sigma_i) / Z."""

    def update_spin(carry: Tuple[jnp.ndarray, ArrayLike], i: int) -> Tuple[Tuple[jnp.ndarray, ArrayLike], None]:
        spins, key = carry
        local_field = jnp.dot(J[i], spins) / jnp.sqrt(spins.shape[0]) + h
        prob_up = jax.nn.sigmoid(2.0 * beta * local_field)

        key, subkey = random.split(key)
        new_spin = lax.select(random.uniform(subkey) < prob_up, 1.0, -1.0)
        spins = spins.at[i].set(new_spin)

        return (spins, key), None

    (spins, _), _ = lax.scan(update_spin, (spins, key), jnp.arange(spins.shape[0]))
    return spins


# ============================================================================
# Vectorized MCMC
# ============================================================================

@partial(jit, static_argnames=('n_steps', 'n_samples', 'method'))
def run_mcmc(
        key: ArrayLike,
        J: jnp.ndarray,
        beta: float,
        h: float = 0.0,
        n_steps: int = 1000,
        n_samples: int = 1000,
        method: Literal['metropolis', 'gibbs', 'metropolis_sweep'] = 'metropolis'
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run MCMC with n_samples independent replicas (fully vectorized).

    Args:
        key: PRNG key
        J: Coupling matrix (n_spins, n_spins)
        beta: Inverse temperature
        h: External field
        n_steps: Monte Carlo steps (single-flip for metropolis, full sweep for gibbs/metropolis_sweep)
        n_samples: Number of parallel chains
        method: 'metropolis' (single-flip), 'gibbs' (full sweep), or 'metropolis_sweep' (N flips per step)

    Returns:
        (final_spins, energy_trajectory):
            - final_spins: (n_samples, n_spins)
            - energy_trajectory: (n_samples, n_steps)
    """
    from parisijax.core.hamiltonian import random_spins, sk_energy

    n_spins = J.shape[0]
    if method == 'metropolis':
        step_fn = _metropolis_step
    elif method == 'metropolis_sweep':
        step_fn = _metropolis_sweep
    else:
        step_fn = _gibbs_step

    # Initialize
    key, subkey = random.split(key)
    init_spins = random_spins(subkey, n_spins, n_samples)

    def mcmc_step(carry: Tuple[jnp.ndarray, ArrayLike], _) -> Tuple[Tuple[jnp.ndarray, ArrayLike], jnp.ndarray]:
        spins, key = carry

        # Update all replicas
        key, *subkeys = random.split(key, n_samples + 1)
        subkeys = jnp.array(subkeys)
        spins = vmap(step_fn, in_axes=(0, 0, None, None, None))(subkeys, spins, J, beta, h)

        # Compute energies
        energies = vmap(sk_energy, in_axes=(0, None, None))(spins, J, h)

        return (spins, key), energies

    (final_spins, _), energy_traj = lax.scan(mcmc_step, (init_spins, key), None, length=n_steps)

    return final_spins, energy_traj.T  # Transpose to (n_samples, n_steps)


# ============================================================================
# Parallel Tempering
# ============================================================================

@jit
def parallel_tempering_step(
        key: ArrayLike,
        spins_all_temps: jnp.ndarray,
        J: jnp.ndarray,
        betas: jnp.ndarray,
        h: float = 0.0
) -> jnp.ndarray:
    """Single PT step: MCMC updates + replica exchange swaps.

    Args:
        key: PRNG key
        spins_all_temps: Configurations (n_temps, n_spins)
        J: Couplings
        betas: Temperature ladder (n_temps,)
        h: External field

    Returns:
        Updated configurations (n_temps, n_spins)
    """
    from parisijax.core.hamiltonian import sk_energy

    n_temps = len(betas)

    # MCMC updates
    key, *subkeys = random.split(key, n_temps + 1)
    spins = vmap(_metropolis_step, in_axes=(0, 0, None, 0, None))(
        jnp.array(subkeys), spins_all_temps, J, betas, h
    )

    # Replica swaps
    def propose_swap(carry: Tuple[jnp.ndarray, ArrayLike], i: int) -> Tuple[Tuple[jnp.ndarray, ArrayLike], float]:
        spins, key = carry

        # Acceptance: exp[(beta_i - beta_{i+1})(E_i - E_{i+1})]
        E_i = sk_energy(spins[i], J, h)
        E_ip1 = sk_energy(spins[i + 1], J, h)
        log_prob = (betas[i] - betas[i + 1]) * (E_i - E_ip1)

        key, subkey = random.split(key)
        accept = random.uniform(subkey) < jnp.exp(jnp.minimum(0.0, log_prob))

        # Swap configurations
        spins_i_new = lax.select(accept, spins[i + 1], spins[i])
        spins_ip1_new = lax.select(accept, spins[i], spins[i + 1])

        spins = spins.at[i].set(spins_i_new)
        spins = spins.at[i + 1].set(spins_ip1_new)

        return (spins, key), accept

    key, subkey = random.split(key)
    (spins, _), _ = lax.scan(propose_swap, (spins, subkey), jnp.arange(n_temps - 1))

    return spins
