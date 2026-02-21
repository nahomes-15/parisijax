"""Tests for the SK Hamiltonian module."""

import jax
import jax.numpy as jnp
import pytest

from parisijax.core.hamiltonian import random_spins, sample_couplings, sk_energy


class TestCouplings:
    """Tests for sample_couplings."""

    def test_symmetry(self, rng):
        J = sample_couplings(rng, 32, 1)[0]
        assert jnp.allclose(J, J.T), "Coupling matrix must be symmetric"

    def test_zero_diagonal(self, rng):
        J = sample_couplings(rng, 32, 1)[0]
        assert jnp.allclose(jnp.diag(J), 0.0), "Diagonal must be zero"

    def test_shape(self, rng):
        n, m = 20, 3
        J = sample_couplings(rng, n, m)
        assert J.shape == (m, n, n)

    def test_variance(self, rng):
        """Off-diagonal elements should have variance ~ 1/2 (symmetrized Gaussians)."""
        J = sample_couplings(rng, 200, 1)[0]
        off_diag = J[jnp.triu_indices(200, k=1)]
        # Variance of (X+X')/2 where X,X' ~ N(0,1) is 1/2
        assert abs(float(jnp.var(off_diag)) - 0.5) < 0.1

    def test_invalid_n_spins(self):
        with pytest.raises(ValueError, match="n_spins must be positive"):
            sample_couplings(jax.random.PRNGKey(0), 0, 1)

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError, match="n_samples must be positive"):
            sample_couplings(jax.random.PRNGKey(0), 10, 0)


class TestSpins:
    """Tests for random_spins."""

    def test_values(self, rng):
        spins = random_spins(rng, 100, 10)
        unique = jnp.unique(spins)
        assert jnp.allclose(jnp.sort(unique), jnp.array([-1.0, 1.0]))

    def test_shape(self, rng):
        spins = random_spins(rng, 50, 7)
        assert spins.shape == (7, 50)

    def test_mean_balance(self, rng):
        """Large sample should be roughly balanced."""
        spins = random_spins(rng, 10000, 1)
        assert abs(float(jnp.mean(spins))) < 0.05


class TestEnergy:
    """Tests for sk_energy."""

    def test_sign_flip_symmetry(self, small_system):
        """E(-sigma) should equal E(sigma) at h=0."""
        J, spins = small_system['J'], small_system['spins']
        e_plus = sk_energy(spins, J, h=0.0)
        e_minus = sk_energy(-spins, J, h=0.0)
        assert jnp.allclose(e_plus, e_minus, atol=1e-5)

    def test_self_consistency(self, rng):
        """Manual energy calculation should match sk_energy."""
        n = 8
        J = sample_couplings(rng, n, 1)[0]
        key2 = jax.random.split(rng)[0]
        spins = random_spins(key2, n, 1)[0]

        # Manual calculation
        interaction = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                interaction += J[i, j] * spins[i] * spins[j]
        manual_e = -interaction / jnp.sqrt(n)

        computed_e = sk_energy(spins, J, h=0.0)
        assert jnp.allclose(manual_e, computed_e, atol=1e-4)
