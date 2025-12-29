"""Sherrington-Kirkpatrick Hamiltonian: H = -(1/√N) Σᵢⱼ Jᵢⱼ σᵢσⱼ - h Σᵢ σᵢ"""

from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import random, jit, vmap
from jax.typing import ArrayLike


@partial(jit, static_argnums=(1, 2))
def sample_couplings(key: ArrayLike, n_spins: int, n_samples: int = 1) -> jnp.ndarray:
    """Generate symmetric SK coupling matrices with J_ij ~ N(0, 1).

    Args:
        key: PRNG key
        n_spins: System size N (must be > 0)
        n_samples: Number of disorder realizations (must be > 0)

    Returns:
        Couplings J of shape (n_samples, n_spins, n_spins), symmetric with zero diagonal

    Raises:
        ValueError: If n_spins <= 0 or n_samples <= 0
    """
    if n_spins <= 0:
        raise ValueError(f"n_spins must be positive, got {n_spins}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    J = random.normal(key, (n_samples, n_spins, n_spins))
    J = (J + jnp.swapaxes(J, -2, -1)) * 0.5  # Symmetrize
    return jnp.where(jnp.eye(n_spins, dtype=bool), 0.0, J)  # Zero diagonal


@jit
def sk_energy(spins: jnp.ndarray, J: jnp.ndarray, h: float = 0.0) -> jnp.ndarray:
    """Compute SK energy: H = -(1/√N) Σᵢⱼ Jᵢⱼ σᵢσⱼ - h Σᵢ σᵢ

    Vmappable over leading batch dimensions.

    Args:
        spins: Configurations ∈ {-1,+1}^N, shape (..., n_spins)
        J: Symmetric couplings, shape (..., n_spins, n_spins)
        h: Uniform external field

    Returns:
        Energy H, shape matches batch dims
    """
    n_spins = spins.shape[-1]
    interaction = -jnp.einsum('...i,...ij,...j->...', spins, J, spins) / (2.0 * jnp.sqrt(n_spins))
    field = -h * jnp.sum(spins, axis=-1)
    return interaction + field


@partial(jit, static_argnums=(1, 2))
def random_spins(key: ArrayLike, n_spins: int, n_samples: int = 1) -> jnp.ndarray:
    """Sample random ±1 spin configurations.

    Args:
        key: PRNG key
        n_spins: System size (must be > 0)
        n_samples: Number of configurations (must be > 0)

    Returns:
        Spins of shape (n_samples, n_spins) with values in {-1, +1}

    Raises:
        ValueError: If n_spins <= 0 or n_samples <= 0
    """
    if n_spins <= 0:
        raise ValueError(f"n_spins must be positive, got {n_spins}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    return jnp.where(random.uniform(key, (n_samples, n_spins)) < 0.5, -1.0, 1.0)
