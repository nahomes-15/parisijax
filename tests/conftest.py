"""Shared fixtures for ParisiJax tests."""

import jax
import pytest

from parisijax.core.hamiltonian import random_spins, sample_couplings


@pytest.fixture
def rng():
    """Deterministic PRNG key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_system(rng):
    """Small (N=16) SK system for fast tests."""
    n_spins = 16
    key1, key2 = jax.random.split(rng)
    J = sample_couplings(key1, n_spins, 1)[0]
    spins = random_spins(key2, n_spins, 1)[0]
    return {'J': J, 'spins': spins, 'n_spins': n_spins}


@pytest.fixture
def medium_system(rng):
    """Medium (N=64) SK system."""
    n_spins = 64
    key1, key2 = jax.random.split(rng)
    J = sample_couplings(key1, n_spins, 1)[0]
    spins = random_spins(key2, n_spins, 50)[0]
    return {'J': J, 'spins': spins, 'n_spins': n_spins}
