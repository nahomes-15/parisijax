"""Overlap distribution and replica structure analysis."""

from typing import Tuple

import jax.numpy as jnp
from jax import jit, random, vmap
from jax.typing import ArrayLike


@jit
def compute_overlap(spins1: jnp.ndarray, spins2: jnp.ndarray) -> jnp.ndarray:
    """Overlap q = N⁻¹ Σᵢ σᵢ⁽¹⁾ σᵢ⁽²⁾ ∈ [-1, 1].

    Vmappable over leading batch dimensions.

    Args:
        spins1, spins2: Configurations (..., n_spins)

    Returns:
        Overlap q, shape matches batch dims
    """
    return jnp.mean(spins1 * spins2, axis=-1)


def sample_overlap_distribution(
    key: ArrayLike,
    J: jnp.ndarray,
    beta: float,
    n_samples: int = 1000,
    n_steps: int = 5000,
    burnin: int = 1000,
    h: float = 0.0
) -> jnp.ndarray:
    """Sample P(q) by running 2n independent MCMC chains and computing pairwise overlaps.

    Args:
        key: PRNG key
        J: Couplings
        beta: Inverse temperature
        n_samples: Number of overlap samples
        n_steps: MCMC steps per chain
        burnin: Thermalization steps to discard
        h: External field

    Returns:
        Overlap samples (n_samples,)
    """
    from parisijax.core.mcmc import run_mcmc

    # Run paired chains
    key, subkey = random.split(key)
    final_spins, _ = run_mcmc(
        subkey, J, beta, h=h,
        n_steps=n_steps + burnin,
        n_samples=2 * n_samples,
        method='metropolis'
    )

    # Compute overlaps between pairs
    spins1, spins2 = final_spins[::2], final_spins[1::2]
    return vmap(compute_overlap)(spins1, spins2)


@jit
def theoretical_overlap_distribution(
    q_parisi: jnp.ndarray,
    m_parisi: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """P(q) from Parisi solution: piecewise constant with jumps at qᵣ.

    Args:
        q_parisi: Overlap breaks (k,)
        m_parisi: Replica indices (k,)

    Returns:
        (q_values, probabilities): Support and weights of P(q)
    """
    q_vals = jnp.concatenate([jnp.array([0.0]), q_parisi])
    delta_m = jnp.diff(jnp.concatenate([jnp.array([0.0]), m_parisi]))
    probs = delta_m / m_parisi[-1]

    return q_vals, probs


@jit
def compute_edwards_anderson_parameter(overlaps: jnp.ndarray) -> float:
    """Edwards-Anderson order parameter: qₑₐ = ⟨q²⟩.

    Args:
        overlaps: Overlap samples

    Returns:
        qₑₐ ∈ [0, 1]
    """
    return jnp.mean(overlaps ** 2)


@jit
def compute_replica_overlap_matrix(spins_batch: jnp.ndarray) -> jnp.ndarray:
    """All pairwise overlaps: Qᵅᵝ = N⁻¹ Σᵢ σᵢ⁽ᵅ⁾ σᵢ⁽ᵝ⁾.

    Args:
        spins_batch: Configurations (n_configs, n_spins)

    Returns:
        Overlap matrix (n_configs, n_configs)
    """
    return jnp.einsum('in,jn->ij', spins_batch, spins_batch) / spins_batch.shape[1]
