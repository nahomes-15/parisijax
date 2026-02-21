"""Tests for the finite-size scaling module."""

import jax.numpy as jnp
import numpy as np

from parisijax.analysis.scaling import (
    binder_cumulant,
    correlation_length_ratio,
    susceptibility,
)


class TestBinderCumulant:
    """Tests for Binder cumulant g = (1/2)(3 - <q^4>/<q^2>^2)."""

    def test_range(self):
        """Binder cumulant should be near 0 for Gaussian and near 1 for delta."""
        # For Gaussian overlaps, g ~ 0 (can be slightly negative from finite sample)
        rng = np.random.RandomState(42)
        overlaps = jnp.array(rng.randn(1000) * 0.3)
        g = float(binder_cumulant(overlaps))
        assert -0.1 <= g <= 1.0 + 1e-6, f"g = {g}"

    def test_gaussian_limit(self):
        """For Gaussian-distributed overlaps, g -> 0."""
        rng = np.random.RandomState(0)
        overlaps = jnp.array(rng.randn(10000))
        g = float(binder_cumulant(overlaps))
        assert abs(g) < 0.1, f"g = {g}, expected ~0 for Gaussian"

    def test_delta_limit(self):
        """For delta-function P(q) at some q0, g -> 1."""
        overlaps = jnp.ones(1000) * 0.7
        g = float(binder_cumulant(overlaps))
        assert abs(g - 1.0) < 0.01, f"g = {g}, expected ~1 for delta"


class TestSusceptibility:
    """Tests for spin glass susceptibility chi_SG = N * <q^2>."""

    def test_scales_with_n(self):
        """chi_SG should scale linearly with N for fixed <q^2>."""
        overlaps = jnp.array([0.5, -0.5, 0.3, -0.3, 0.4])
        chi_16 = float(susceptibility(overlaps, 16))
        chi_64 = float(susceptibility(overlaps, 64))
        assert abs(chi_64 / chi_16 - 4.0) < 1e-4

    def test_positive(self):
        """Susceptibility should always be non-negative."""
        rng = np.random.RandomState(1)
        overlaps = jnp.array(rng.randn(100) * 0.5)
        chi = float(susceptibility(overlaps, 32))
        assert chi >= 0.0


class TestCorrelationLengthRatio:
    """Tests for xi/L proxy."""

    def test_finite(self):
        rng = np.random.RandomState(2)
        overlaps = jnp.array(rng.randn(500) * 0.3)
        xi = float(correlation_length_ratio(overlaps))
        assert np.isfinite(xi)
        assert xi >= 0.0
