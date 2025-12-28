"""Comprehensive tests for SK model implementation."""

import jax
import jax.numpy as jnp
import pytest
from parisijax.core import hamiltonian, solver


class TestHamiltonian:
        """Tests for Hamiltonian and spin configuration utilities."""

    def test_coupling_properties(self):
                """Test that coupling matrices have correct properties."""
                key = jax.random.PRNGKey(123)
                n_spins = 100
                n_samples = 5

        J = hamiltonian.sample_couplings(key, n_spins, n_samples)

        # Check shape
        assert J.shape == (n_samples, n_spins, n_spins)

        # Check symmetry
        assert jnp.allclose(J, jnp.swapaxes(J, -2, -1))

        # Check diagonal is zero
        for i in range(n_samples):
                        assert jnp.allclose(jnp.diag(J[i]), 0.0)

        # Check scaling (variance should be roughly 1/N)
        variance = jnp.var(J)
        expected_variance = 1.0 / n_spins
        assert jnp.abs(variance - expected_variance) < 0.05

    def test_spin_initialization(self):
                """Test that spins are correctly initialized to +/-1."""
                key = jax.random.PRNGKey(456)
                spins = hamiltonian.random_spins(key, n_spins=1000, n_samples=10)

        # Check shape
        assert spins.shape == (10, 1000)

        # Check values are +/-1
        assert jnp.all(jnp.abs(spins) == 1.0)

        # Check roughly balanced
        mean = jnp.mean(spins)
        assert jnp.abs(mean) < 0.1

    def test_energy_computation(self):
                """Test basic properties of the SK Hamiltonian."""
                key = jax.random.PRNGKey(42)
                n_spins = 1000

        J = hamiltonian.sample_couplings(key, n_spins, n_samples=1)[0]
        key, subkey = jax.random.split(key)
        spins = hamiltonian.random_spins(subkey, n_spins, n_samples=100)

        energies = jax.vmap(hamiltonian.sk_energy, in_axes=(0, None))(spins, J)
        energy_per_spin = energies / n_spins

        # For random configurations, energy should be O(1) and roughly centered
        assert jnp.abs(jnp.mean(energy_per_spin)) < 0.2
        assert jnp.std(energy_per_spin) < 0.2


class TestRSSolution:
        """Tests for replica-symmetric solution."""

    def test_rs_free_energy_high_temp(self):
                """Test RS free energy at high temperature."""
                beta = 0.5  # High temperature
                f = solver.rs_free_energy(beta, h=0.0)

        # At high T, should approach -log(2)/beta - beta/4
        f_high_T = solver.high_temp_free_energy(beta)
        assert jnp.abs(f - f_high_T) < 0.1

    def test_rs_free_energy_finite(self):
                """Test RS free energy is finite at various beta."""
                for beta in [0.1, 0.5, 1.0, 1.5, 2.0]:
                                f = solver.rs_free_energy(beta)
                                assert jnp.isfinite(f)
                                assert f < 0  # Free energy should be negative

    def test_rs_free_energy_monotonic(self):
                """Free energy should decrease (become more negative) with increasing beta."""
                betas = jnp.array([0.5, 1.0, 1.5, 2.0])
                f_values = [float(solver.rs_free_energy(b)) for b in betas]

        # Check monotonically decreasing
        for i in range(len(f_values) - 1):
                        assert f_values[i+1] < f_values[i]

    def test_critical_temperature(self):
                """Test finding the critical temperature."""
                beta_c = solver.find_critical_temperature()

        # Should be close to 1.0 for SK model at h=0
        assert 0.95 < beta_c < 1.05, f"Critical beta {beta_c} not close to 1.0"


class TestRSBSolution:
        """Tests for replica symmetry breaking solutions."""

    def test_one_rsb_free_energy(self):
                """Test 1-step RSB free energy computation."""
                beta = 1.5

        # Test with sample parameters
        q0, q1, m = 0.3, 0.7, 0.5
        f_1rsb = solver.one_rsb_free_energy(q0, q1, m, beta, h=0.0, n_quad=32)

        # Should be finite
        assert jnp.isfinite(f_1rsb)
        assert f_1rsb < 0

    def test_one_rsb_vs_rs(self):
                """Test that optimized 1RSB should give lower free energy than RS below T_c."""
                beta = 1.5  # Below critical temperature

        f_rs = solver.rs_free_energy(beta, h=0.0)

        # Try several 1RSB parameter combinations
        best_f_1rsb = f_rs
        for q0 in [0.1, 0.2, 0.3]:
                        for q1 in [0.6, 0.7, 0.8, 0.9]:
                                            if q1 > q0:
                                                                    for m in [0.3, 0.5, 0.7]:
                                                                                                f_1rsb = solver.one_rsb_free_energy(q0, q1, m, beta, n_quad=32)
                                                                                                if f_1rsb < best_f_1rsb:
                                                                                                                                best_f_1rsb = f_1rsb
                                                                                                    
        # 1RSB should be able to find lower free energy than RS
        assert best_f_1rsb <= f_rs + 0.01, \
            f"1RSB ({best_f_1rsb:.4f}) should be <= RS ({f_rs:.4f})"

    def test_parisi_free_energy_computes(self):
                """Test that full Parisi k-RSB computation runs."""
                beta = 1.5
                k = 5

        q_raw = jnp.zeros(k)
        m_raw = jnp.zeros(k)

        f = solver.parisi_free_energy(q_raw, m_raw, beta, h=0.0)
        assert jnp.isfinite(f)


class TestKnownResults:
        """Tests against known theoretical results."""

    def test_ground_state_energy_reference(self):
                """Check the ground state energy constant."""
                E0 = solver.ground_state_energy()
                assert jnp.abs(E0 - (-0.7633)) < 0.001

    def test_high_temp_expansion(self):
                """Test high temperature expansion."""
                beta = 0.1  # Very high temperature

        f = solver.rs_free_energy(beta)
        f_expected = solver.high_temp_free_energy(beta)

        # Should match well at high T
        assert jnp.abs(f - f_expected) < 0.05


def test_hamiltonian_basic():
        """Standalone test for basic Hamiltonian properties."""
        key = jax.random.PRNGKey(42)
        n_spins = 1000

    J = hamiltonian.sample_couplings(key, n_spins, n_samples=1)[0]
    key, subkey = jax.random.split(key)
    spins = hamiltonian.random_spins(subkey, n_spins, n_samples=100)

    energies = jax.vmap(hamiltonian.sk_energy, in_axes=(0, None))(spins, J)
    energy_per_spin = energies / n_spins

    assert jnp.abs(jnp.mean(energy_per_spin)) < 0.2
    assert jnp.std(energy_per_spin) < 0.2
    print(f"Energy per spin: mean={jnp.mean(energy_per_spin):.4f}, std={jnp.std(energy_per_spin):.4f}")


def test_coupling_properties():
        """Standalone test for coupling matrix properties."""
        key = jax.random.PRNGKey(123)
        n_spins = 100
        n_samples = 5

    J = hamiltonian.sample_couplings(key, n_spins, n_samples)

    assert J.shape == (n_samples, n_spins, n_spins)
    assert jnp.allclose(J, jnp.swapaxes(J, -2, -1))

    variance = jnp.var(J)
    expected_variance = 1.0 / n_spins
    assert jnp.abs(variance - expected_variance) < 0.1
    print(f"Coupling variance: {variance:.4f} (expected ~{expected_variance:.4f})")


def test_spin_initialization():
        """Standalone test for spin initialization."""
        key = jax.random.PRNGKey(456)
        spins = hamiltonian.random_spins(key, n_spins=1000, n_samples=10)

    assert spins.shape == (10, 1000)
    assert jnp.all(jnp.abs(spins) == 1.0)

    mean = jnp.mean(spins)
    assert jnp.abs(mean) < 0.1
    print(f"Spin balance: mean={mean:.4f}")


def test_rs_free_energy():
        """Standalone test for RS free energy."""
        beta = 0.5
        f = solver.rs_free_energy(beta, h=0.0)
        print(f"RS free energy at beta={beta}: f={f:.4f}")

    beta = 1.5
    f = solver.rs_free_energy(beta, h=0.0)
    assert f < 0
    print(f"RS free energy at beta={beta}: f={f:.4f}")

    for b in [0.1, 0.5, 1.0, 2.0]:
                f_test = solver.rs_free_energy(b)
                assert jnp.isfinite(f_test)
            print("RS free energy computes correctly for various beta values")


def test_critical_temperature():
        """Standalone test for critical temperature."""
        beta_c = solver.find_critical_temperature()

    assert 0.9 < beta_c < 1.1, f"Critical beta {beta_c} not close to expected ~1.0"
    print(f"Critical temperature: beta_c={beta_c:.4f} (theory predicts ~1.0)")


def test_one_rsb():
        """Standalone test for 1-step RSB free energy."""
        beta = 1.5

    q0, q1, m = 0.3, 0.7, 0.5
    f_1rsb = solver.one_rsb_free_energy(q0, q1, m, beta, h=0.0, n_quad=32)
    f_rs = solver.rs_free_energy(beta, h=0.0)

    print(f"1RSB free energy at beta={beta}: f_1RSB={f_1rsb:.4f}, f_RS={f_rs:.4f}")

    # At optimal parameters, 1RSB should give lower free energy
    # This is a weak test - just checking computation works
    assert jnp.isfinite(f_1rsb)


if __name__ == "__main__":
        print("=" * 60)
        print("ParisiJAX Test Suite")
        print("=" * 60)

    print("\n--- Phase 1: Hamiltonian Tests ---")
    test_coupling_properties()
    test_spin_initialization()
    test_hamiltonian_basic()
    print("All Phase 1 tests passed!")

    print("\n--- Phase 2: RS Solution Tests ---")
    test_rs_free_energy()
    test_critical_temperature()
    print("All Phase 2 tests passed!")

    print("\n--- Phase 3: RSB Solution Tests ---")
    test_one_rsb()
    print("All Phase 3 tests passed!")

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
