"""Tests for the MCMC module."""

import jax
import jax.numpy as jnp

from parisijax.core.hamiltonian import random_spins, sample_couplings
from parisijax.core.mcmc import (
    _metropolis_step,
    _metropolis_sweep,
    parallel_tempering_step,
    run_mcmc,
)


class TestMetropolis:
    """Metropolis sampler tests."""

    def test_output_shape(self, rng, small_system):
        J, n = small_system['J'], small_system['n_spins']
        key1, key2 = jax.random.split(rng)
        spins = random_spins(key1, n, 1)[0]
        result = _metropolis_step(key2, spins, J, 1.0, 0.0)
        assert result.shape == (n,)

    def test_spin_values(self, rng, small_system):
        """Output spins must be in {-1, +1}."""
        J, n = small_system['J'], small_system['n_spins']
        key1, key2 = jax.random.split(rng)
        spins = random_spins(key1, n, 1)[0]
        result = _metropolis_step(key2, spins, J, 1.0, 0.0)
        assert jnp.all((result == 1.0) | (result == -1.0))

    def test_high_temp_acceptance(self, rng):
        """At beta=0, every flip should be accepted (dE * 0 = 0, exp(0) = 1)."""
        n = 32
        key1, key2 = jax.random.split(rng)
        J = sample_couplings(key1, n, 1)[0]
        spins = random_spins(key2, n, 1)[0]

        # Run many steps at beta=0, count changes
        n_flips = 0
        current = spins
        key = jax.random.PRNGKey(99)
        for _ in range(200):
            key, subkey = jax.random.split(key)
            new = _metropolis_step(subkey, current, J, 0.0, 0.0)
            n_flips += int(jnp.sum(current != new))
            current = new

        # At beta=0, acceptance = 100%, so each step flips exactly 1 spin
        assert n_flips == 200


class TestMetropolisSweep:
    """Tests for the full-sweep Metropolis."""

    def test_output_shape(self, rng, small_system):
        J, n = small_system['J'], small_system['n_spins']
        spins = random_spins(rng, n, 1)[0]
        result = _metropolis_sweep(jax.random.split(rng)[0], spins, J, 1.0, 0.0)
        assert result.shape == (n,)

    def test_spin_values(self, rng, small_system):
        J, n = small_system['J'], small_system['n_spins']
        spins = random_spins(rng, n, 1)[0]
        result = _metropolis_sweep(jax.random.split(rng)[0], spins, J, 1.0, 0.0)
        assert jnp.all((result == 1.0) | (result == -1.0))


class TestRunMCMC:
    """Tests for vectorized MCMC."""

    def test_output_shapes(self, rng, small_system):
        J, n = small_system['J'], small_system['n_spins']
        n_samples, n_steps = 10, 50
        spins, energies = run_mcmc(rng, J, 1.0, n_steps=n_steps, n_samples=n_samples)
        assert spins.shape == (n_samples, n)
        assert energies.shape == (n_samples, n_steps)

    def test_energy_decreases_low_temp(self, rng):
        """At low temperature, average energy should decrease over MCMC steps."""
        n = 32
        J = sample_couplings(rng, n, 1)[0]
        key = jax.random.split(rng)[0]
        _, energies = run_mcmc(key, J, beta=5.0, n_steps=500, n_samples=20)
        early_mean = float(jnp.mean(energies[:, :50]))
        late_mean = float(jnp.mean(energies[:, -50:]))
        assert late_mean < early_mean

    def test_metropolis_sweep_method(self, rng, small_system):
        """The 'metropolis_sweep' method should run without error."""
        J, n = small_system['J'], small_system['n_spins']
        spins, energies = run_mcmc(
            rng, J, 1.0, n_steps=20, n_samples=5, method='metropolis_sweep'
        )
        assert spins.shape == (5, n)


class TestParallelTempering:
    """Tests for parallel tempering."""

    def test_output_shape(self, rng, small_system):
        J, n = small_system['J'], small_system['n_spins']
        betas = jnp.array([0.5, 1.0, 2.0, 4.0])
        n_temps = len(betas)
        init_spins = random_spins(rng, n, n_temps)
        result = parallel_tempering_step(
            jax.random.split(rng)[0], init_spins, J, betas
        )
        assert result.shape == (n_temps, n)

    def test_spin_values_preserved(self, rng, small_system):
        """After PT step, all spins should still be in {-1, +1}."""
        J, n = small_system['J'], small_system['n_spins']
        betas = jnp.array([0.5, 1.0, 2.0])
        init_spins = random_spins(rng, n, 3)
        result = parallel_tempering_step(
            jax.random.split(rng)[0], init_spins, J, betas
        )
        assert jnp.all((result == 1.0) | (result == -1.0))
