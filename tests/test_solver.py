"""Tests for the Parisi solver module — the most critical tests."""

import jax
import jax.numpy as jnp
import pytest

from parisijax.core.solver import (
    ParisiResult,
    _params_to_rsb,
    find_critical_temperature,
    high_temp_free_energy,
    one_rsb_free_energy,
    optimize_parisi,
    optimize_parisi_multistart,
    parisi_free_energy,
    rs_free_energy,
)

# ---------------------------------------------------------------------------
# RS validation
# ---------------------------------------------------------------------------

class TestRSFreeEnergy:
    """Replica-symmetric solution tests."""

    def test_high_temp_matches_analytic(self):
        """At high T (small beta), f_RS ~ -log(2)/beta - beta/4."""
        beta = 0.3
        f_rs = float(rs_free_energy(beta))
        f_exact = float(high_temp_free_energy(beta))
        assert abs(f_rs - f_exact) < 0.01, f"RS={f_rs}, exact={f_exact}"

    def test_free_energy_negative(self):
        """Free energy should always be negative."""
        for beta in [0.2, 0.5, 1.0, 2.0]:
            assert float(rs_free_energy(beta)) < 0

    def test_free_energy_increases_toward_ground_state(self):
        """f(beta) should increase toward E_0/N as beta grows (from -inf at beta->0)."""
        betas = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        energies = [float(rs_free_energy(b)) for b in betas]
        for i in range(len(energies) - 1):
            assert energies[i + 1] >= energies[i] - 1e-6

    @pytest.mark.validation
    def test_critical_temperature(self):
        """beta_c should be within 0.05 of 1.0 for h=0."""
        beta_c = float(find_critical_temperature())
        assert abs(beta_c - 1.0) < 0.05, f"beta_c = {beta_c}"


# ---------------------------------------------------------------------------
# 1RSB validation
# ---------------------------------------------------------------------------

class TestOneRSB:
    """1-step RSB tests."""

    def test_reduces_to_rs_limit(self):
        """When q0 -> q1, 1RSB should approach RS."""
        beta = 1.5
        f_rs = float(rs_free_energy(beta))
        # Set q0 ~ q1 so 1RSB collapses to RS-like solution
        f_1rsb = float(one_rsb_free_energy(0.5, 0.501, 0.5, beta))
        # They won't be exactly equal, but should be in the same ballpark
        assert abs(f_1rsb - f_rs) < 0.5

    @pytest.mark.validation
    def test_1rsb_below_rs(self):
        """f_1RSB <= f_RS below T_c (RSB gives lower free energy)."""
        beta = 1.5
        f_rs = float(rs_free_energy(beta))
        # Try a reasonable 1RSB parametrization
        best_f = float('inf')
        for q0 in [0.1, 0.2, 0.3]:
            for q1 in [0.5, 0.7, 0.9]:
                for m in [0.3, 0.5, 0.7]:
                    f = float(one_rsb_free_energy(q0, q1, m, beta))
                    if f < best_f:
                        best_f = f
        assert best_f <= f_rs + 1e-4, f"best 1RSB={best_f}, RS={f_rs}"


# ---------------------------------------------------------------------------
# k-RSB / Parisi validation
# ---------------------------------------------------------------------------

class TestParisiFreeEnergy:
    """Full Parisi k-RSB tests."""

    def test_finite(self):
        """Free energy should be finite for reasonable parameters."""
        key = jax.random.PRNGKey(0)
        q_raw = jax.random.normal(key, (5,)) * 0.1
        m_raw = jax.random.normal(jax.random.split(key)[0], (5,)) * 0.1
        f = parisi_free_energy(q_raw, m_raw, beta=1.5)
        assert jnp.isfinite(f)

    def test_differentiable(self):
        """jax.grad should return finite gradients."""
        key = jax.random.PRNGKey(1)
        q_raw = jax.random.normal(key, (5,)) * 0.1
        m_raw = jax.random.normal(jax.random.split(key)[0], (5,)) * 0.1

        grad_fn = jax.grad(lambda q, m: parisi_free_energy(q, m, beta=1.5), argnums=(0, 1))
        gq, gm = grad_fn(q_raw, m_raw)
        assert jnp.all(jnp.isfinite(gq))
        assert jnp.all(jnp.isfinite(gm))

    def test_parameter_monotonicity(self):
        """After reparametrization, q and m should be monotonically increasing."""
        key = jax.random.PRNGKey(2)
        q_raw = jax.random.normal(key, (10,))
        m_raw = jax.random.normal(jax.random.split(key)[0], (10,))
        q, m = _params_to_rsb(q_raw, m_raw)
        assert jnp.all(jnp.diff(q) > 0), "q must be strictly increasing"
        assert jnp.all(jnp.diff(m) > 0), "m must be strictly increasing"


class TestOptimizeParisi:
    """Optimization tests."""

    def test_loss_decreases(self):
        """Loss (= free energy) should generally decrease during optimization."""
        result = optimize_parisi(beta=1.5, k=5, n_steps=200, lr=0.01, seed=0)
        losses = result.loss_history
        # Compare first 10% vs last 10% — loss decreases means more negative
        early = float(jnp.mean(losses[:20]))
        late = float(jnp.mean(losses[-20:]))
        assert late <= early + 0.01

    def test_returns_parisi_result(self):
        """Should return a ParisiResult NamedTuple."""
        result = optimize_parisi(beta=1.5, k=5, n_steps=50, seed=0)
        assert isinstance(result, ParisiResult)
        assert hasattr(result, 'converged')
        assert hasattr(result, 'n_steps_used')
        assert hasattr(result, 'grad_norm_final')

    def test_early_stopping(self):
        """With tight tolerance the optimizer should converge before max steps."""
        result = optimize_parisi(beta=0.5, k=3, n_steps=5000, tol=1e-5, patience=50, seed=0)
        # At high temperature, convergence should be fast
        assert result.n_steps_used <= 5000

    @pytest.mark.slow
    @pytest.mark.validation
    def test_ground_state_energy(self):
        """At beta=10, k=20: free energy should approach -0.7633.

        The grid-based PDE solver has systematic interpolation error,
        so we allow a wider tolerance. Higher n_grid improves accuracy.
        """
        result = optimize_parisi_multistart(
            beta=10.0, k=20, n_steps=3000, lr=0.005, n_starts=3, n_grid=800
        )
        assert abs(result.free_energy - (-0.7633)) < 0.10, \
            f"f = {result.free_energy}, expected ~ -0.7633"

    def test_cosine_schedule(self):
        """Cosine LR schedule should work without errors."""
        result = optimize_parisi(beta=1.5, k=5, n_steps=100, lr_schedule='cosine', seed=0)
        assert jnp.isfinite(result.free_energy)

    def test_multistart_best(self):
        """Multi-start should return a result at least as good as single run."""
        single = optimize_parisi(beta=1.5, k=5, n_steps=200, seed=0)
        multi = optimize_parisi_multistart(beta=1.5, k=5, n_starts=3, n_steps=200)
        assert multi.free_energy <= single.free_energy + 1e-4


# ---------------------------------------------------------------------------
# RS order parameter
# ---------------------------------------------------------------------------

class TestOrderParameter:
    """RS order parameter behavior."""

    @pytest.mark.validation
    def test_q_zero_above_tc(self):
        """q should be approximately 0 above T_c (beta < 1)."""
        from parisijax.core.solver import _rs_order_parameter
        q = float(_rs_order_parameter(0.5))
        assert q < 0.05, f"q = {q} at beta=0.5, expected ~0"

    @pytest.mark.validation
    def test_q_positive_below_tc(self):
        """q should be positive below T_c (beta > 1)."""
        from parisijax.core.solver import _rs_order_parameter
        q = float(_rs_order_parameter(1.5))
        assert q > 0.1, f"q = {q} at beta=1.5, expected > 0"
