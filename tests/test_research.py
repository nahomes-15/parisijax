"""Tests for the research module: k-RSB convergence, thermodynamics, AT line."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from parisijax.analysis.research import (
    ConvergenceResult,
    ThermodynamicResult,
    ATLineResult,
    krsb_convergence,
    fit_convergence_rate,
    thermodynamic_quantities,
    compute_at_line,
    rs_vs_rsb_comparison,
    extract_parisi_function,
    _at_eigenvalue,
)
from parisijax.core.solver import rs_free_energy


# ---------------------------------------------------------------------------
# AT Eigenvalue
# ---------------------------------------------------------------------------

class TestATEigenvalue:
    """Tests for the AT stability eigenvalue."""

    def test_stable_above_tc(self):
        """AT eigenvalue should be positive (RS stable) for beta < 1 at h=0."""
        lam = float(_at_eigenvalue(0.5, 0.0, 32))
        assert lam > 0, f"lambda_AT = {lam}, expected > 0 above T_c"

    def test_unstable_below_tc(self):
        """AT eigenvalue should be negative (RS unstable) for beta > 1 at h=0."""
        lam = float(_at_eigenvalue(1.5, 0.0, 32))
        assert lam < 0, f"lambda_AT = {lam}, expected < 0 below T_c"

    def test_finite(self):
        """AT eigenvalue should be finite for all reasonable parameters."""
        for beta in [0.3, 1.0, 2.0, 5.0]:
            for h in [0.0, 0.5, 1.0]:
                lam = _at_eigenvalue(beta, h, 32)
                assert jnp.isfinite(lam), f"Non-finite at beta={beta}, h={h}"

    def test_field_stabilizes(self):
        """External field should stabilize RS (increase AT eigenvalue)."""
        lam_h0 = float(_at_eigenvalue(1.5, 0.0, 32))
        lam_h1 = float(_at_eigenvalue(1.5, 1.0, 32))
        assert lam_h1 > lam_h0, "Field should stabilize RS"


# ---------------------------------------------------------------------------
# AT Line
# ---------------------------------------------------------------------------

class TestATLine:
    """Tests for AT line computation."""

    def test_at_h0(self):
        """AT line at h=0 should give T_c = 1 (beta_c = 1)."""
        result = compute_at_line(
            h_values=np.array([0.0]),
            beta_range=(0.5, 2.0),
            tol=1e-4,
            n_quad=32,
        )
        assert isinstance(result, ATLineResult)
        assert abs(result.beta_c_values[0] - 1.0) < 0.05, \
            f"beta_c(h=0) = {result.beta_c_values[0]}, expected ~1.0"

    def test_at_line_decreasing(self):
        """T_c should decrease with increasing field h."""
        result = compute_at_line(
            h_values=np.array([0.0, 0.3, 0.6, 1.0]),
            beta_range=(0.5, 10.0),
            tol=1e-4,
            n_quad=32,
        )
        finite = np.isfinite(result.beta_c_values)
        # beta_c should increase (T_c decreases) with h
        bc = result.beta_c_values[finite]
        for i in range(len(bc) - 1):
            assert bc[i+1] >= bc[i] - 0.01, \
                f"beta_c should increase with h: {bc}"


# ---------------------------------------------------------------------------
# k-RSB Convergence
# ---------------------------------------------------------------------------

class TestKRSBConvergence:
    """Tests for k-RSB convergence analysis."""

    @pytest.mark.slow
    def test_convergence_basic(self):
        """k-RSB free energies should decrease (improve) with increasing k."""
        result = krsb_convergence(
            beta=1.5, h=0.0,
            k_values=[1, 2, 3, 5],
            n_steps=1000, n_quad=32, n_grid=200, n_starts=1,
        )
        assert isinstance(result, ConvergenceResult)
        assert len(result.free_energies) == 4
        # Free energy should generally decrease with k
        assert result.free_energies[-1] <= result.free_energies[0] + 0.01

    def test_convergence_result_fields(self):
        """ConvergenceResult should have all expected fields."""
        result = krsb_convergence(
            beta=1.5, h=0.0,
            k_values=[1, 2],
            n_steps=200, n_quad=16, n_grid=100, n_starts=1,
        )
        assert result.beta == 1.5
        assert result.h == 0.0
        assert len(result.k_values) == 2
        assert len(result.q_ea_values) == 2

    def test_fit_convergence_rate(self):
        """fit_convergence_rate should return finite values."""
        # Create synthetic convergence data
        k_vals = np.arange(1, 11)
        f_vals = -0.8 + 0.1 / k_vals**2  # Known: alpha=2
        result = ConvergenceResult(
            k_values=k_vals,
            free_energies=f_vals,
            delta_f=np.diff(f_vals, prepend=0),
            q_ea_values=np.ones(10) * 0.5,
            converged_flags=np.ones(10, dtype=bool),
            beta=1.5,
            h=0.0,
        )
        alpha, C, f_inf = fit_convergence_rate(result, k_min=2)
        assert np.isfinite(alpha)
        assert np.isfinite(C)
        # Should recover alpha in reasonable range (exact recovery is hard
        # because f_inf is estimated from f[-1], biasing the log-log fit)
        assert 1.0 < alpha < 5.0, f"alpha={alpha}, expected in (1, 5)"


# ---------------------------------------------------------------------------
# Thermodynamic Quantities
# ---------------------------------------------------------------------------

class TestThermodynamics:
    """Tests for autodiff thermodynamic quantities."""

    @pytest.mark.slow
    def test_basic_computation(self):
        """Should compute finite thermodynamic quantities."""
        result = thermodynamic_quantities(
            beta=1.5, h=0.0, k=5,
            n_quad=32, n_grid=150, n_steps=500,
            dbeta=0.01, dh=0.01,
        )
        assert isinstance(result, ThermodynamicResult)
        assert np.isfinite(result.free_energy)
        assert np.isfinite(result.internal_energy)
        assert np.isfinite(result.entropy)
        assert np.isfinite(result.specific_heat)

    @pytest.mark.slow
    def test_high_temp_magnetization_zero(self):
        """At h=0 magnetization should be ~0 by symmetry."""
        result = thermodynamic_quantities(
            beta=0.5, h=0.0, k=3,
            n_quad=32, n_grid=150, n_steps=500,
            dbeta=0.01, dh=0.01,
        )
        assert abs(result.magnetization) < 0.1, \
            f"M = {result.magnetization}, expected ~0 at h=0"


# ---------------------------------------------------------------------------
# RS vs RSB Comparison
# ---------------------------------------------------------------------------

class TestRSvsRSB:
    """Tests for RS vs RSB free energy comparison."""

    @pytest.mark.slow
    def test_rsb_improves_below_tc(self):
        """RSB free energy should be lower than RS below T_c."""
        result = rs_vs_rsb_comparison(
            betas=np.array([1.5]),
            h=0.0, k=5,
            n_quad=32, n_grid=200, n_steps=1000,
        )
        assert result['delta_f'][0] >= -0.01, \
            "f_RS should be >= f_RSB (delta_f >= 0)"

    @pytest.mark.slow
    def test_rs_exact_above_tc(self):
        """RS and RSB should agree above T_c (gap ~ 0)."""
        result = rs_vs_rsb_comparison(
            betas=np.array([0.5]),
            h=0.0, k=3,
            n_quad=32, n_grid=200, n_steps=1000,
        )
        assert abs(result['delta_f'][0]) < 0.05, \
            f"Gap = {result['delta_f'][0]}, expected ~0 above T_c"


# ---------------------------------------------------------------------------
# Parisi Function Extraction
# ---------------------------------------------------------------------------

class TestParisiFunction:
    """Tests for extracting x(q)."""

    @pytest.mark.slow
    def test_extraction(self):
        """Should return valid q and x arrays."""
        q_vals, x_vals = extract_parisi_function(
            beta=1.5, h=0.0, k=5,
            n_quad=32, n_grid=150, n_steps=500, n_starts=1,
        )
        assert len(q_vals) == 6  # k+1
        assert len(x_vals) == 5  # k
        # q should be monotonically increasing
        assert np.all(np.diff(q_vals) >= -1e-6)
        # x should be monotonically increasing
        assert np.all(np.diff(x_vals) >= -1e-6)
        # All values in [0, 1]
        assert np.all(q_vals >= -1e-6) and np.all(q_vals <= 1.0 + 1e-6)
        assert np.all(x_vals >= -1e-6) and np.all(x_vals <= 1.0 + 1e-6)
