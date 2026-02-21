"""Tests for the overlap analysis module."""

import jax
import jax.numpy as jnp
import pytest

from parisijax.analysis.overlap import (
    compute_edwards_anderson_parameter,
    compute_overlap,
    compute_replica_overlap_matrix,
    theoretical_overlap_distribution,
)
from parisijax.core.hamiltonian import random_spins


class TestComputeOverlap:
    """Tests for pairwise overlap computation."""

    def test_self_overlap(self, rng):
        """Overlap of a configuration with itself should be 1."""
        spins = random_spins(rng, 100, 1)[0]
        q = float(compute_overlap(spins, spins))
        assert abs(q - 1.0) < 1e-5

    def test_antiparallel_overlap(self, rng):
        """Overlap with negated configuration should be -1."""
        spins = random_spins(rng, 100, 1)[0]
        q = float(compute_overlap(spins, -spins))
        assert abs(q - (-1.0)) < 1e-5

    def test_range(self, rng):
        """Overlap should be in [-1, 1]."""
        key1, key2 = jax.random.split(rng)
        s1 = random_spins(key1, 50, 100)
        s2 = random_spins(key2, 50, 100)
        overlaps = jax.vmap(compute_overlap)(s1, s2)
        assert jnp.all(overlaps >= -1.0 - 1e-6)
        assert jnp.all(overlaps <= 1.0 + 1e-6)


class TestOverlapMatrix:
    """Tests for replica overlap matrix."""

    def test_diagonal_is_one(self, rng):
        """Diagonal of overlap matrix should be 1."""
        spins = random_spins(rng, 50, 10)
        Q = compute_replica_overlap_matrix(spins)
        assert jnp.allclose(jnp.diag(Q), 1.0, atol=1e-5)

    def test_symmetric(self, rng):
        """Overlap matrix should be symmetric."""
        spins = random_spins(rng, 50, 10)
        Q = compute_replica_overlap_matrix(spins)
        assert jnp.allclose(Q, Q.T, atol=1e-6)

    def test_shape(self, rng):
        n_configs = 8
        spins = random_spins(rng, 50, n_configs)
        Q = compute_replica_overlap_matrix(spins)
        assert Q.shape == (n_configs, n_configs)


class TestEdwardsAnderson:
    """Tests for EA order parameter."""

    @pytest.mark.validation
    def test_paramagnetic_phase(self, rng):
        """q_EA should be small for random (uncorrelated) overlaps."""
        # Random overlaps at high temperature should be close to 0
        key1, key2 = jax.random.split(rng)
        s1 = random_spins(key1, 200, 500)
        s2 = random_spins(key2, 200, 500)
        overlaps = jax.vmap(compute_overlap)(s1, s2)
        q_EA = float(compute_edwards_anderson_parameter(overlaps))
        # For random spins, <q^2> ~ 1/N
        assert q_EA < 0.05, f"q_EA = {q_EA}, expected ~0 for random overlaps"


class TestTheoreticalDistribution:
    """Tests for P(q) from Parisi solution."""

    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to 1."""
        q = jnp.array([0.2, 0.5, 0.8])
        m = jnp.array([0.3, 0.6, 1.0])
        _, probs = theoretical_overlap_distribution(q, m)
        assert abs(float(jnp.sum(probs)) - 1.0) < 1e-5
