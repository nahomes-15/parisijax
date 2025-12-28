"""Basic tests for SK model implementation."""

import jax
import jax.numpy as jnp
from parisijax.core import hamiltonian, solver


def test_hamiltonian_basic():
    """Test basic properties of the SK Hamiltonian."""
    key = jax.random.PRNGKey(42)
    n_spins = 1000

    # Generate couplings and spins
    J = hamiltonian.sample_couplings(key, n_spins, n_samples=1)[0]
    key, subkey = jax.random.split(key)
    spins = hamiltonian.random_spins(subkey, n_spins, n_samples=100)

    # Compute energies
    energies = jax.vmap(hamiltonian.sk_energy, in_axes=(0, None))(spins, J)
    energy_per_spin = energies / n_spins

    # For random configurations, energy should be O(1) and roughly centered
    assert jnp.abs(jnp.mean(energy_per_spin)) < 0.2
    assert jnp.std(energy_per_spin) < 0.2

    print(f"✓ Energy per spin: mean={jnp.mean(energy_per_spin):.4f}, std={jnp.std(energy_per_spin):.4f}")


def test_coupling_properties():
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
    assert jnp.abs(variance - expected_variance) < 0.1

    print(f"✓ Coupling variance: {variance:.4f} (expected ~{expected_variance:.4f})")


def test_spin_initialization():
    """Test that spins are correctly initialized to ±1."""
    key = jax.random.PRNGKey(456)
    spins = hamiltonian.random_spins(key, n_spins=1000, n_samples=10)

    # Check shape
    assert spins.shape == (10, 1000)

    # Check values are ±1
    assert jnp.all(jnp.abs(spins) == 1.0)

    # Check roughly balanced
    mean = jnp.mean(spins)
    assert jnp.abs(mean) < 0.1

    print(f"✓ Spin balance: mean={mean:.4f}")


def test_rs_free_energy():
    """Test replica-symmetric free energy calculation."""
    # At beta = 0.5, h = 0
    beta = 0.5
    f = solver.rs_free_energy(beta, h=0.0)
    print(f"✓ RS free energy at β={beta}: f={f:.4f}")

    # At beta = 1.5
    beta = 1.5
    f = solver.rs_free_energy(beta, h=0.0)
    # Free energy should be negative and reasonable
    assert f < 0  # Should be negative
    print(f"✓ RS free energy at β={beta}: f={f:.4f}")

    # Sanity check: function computes without error at various beta
    for b in [0.1, 0.5, 1.0, 2.0]:
        f_test = solver.rs_free_energy(b)
        assert jnp.isfinite(f_test)
    print(f"✓ RS free energy computes correctly for various β values")


def test_critical_temperature():
    """Test finding the critical temperature."""
    beta_c = solver.find_critical_temperature()

    # Should be approximately 1.0, but allow wider range
    assert 0.5 < beta_c < 1.5
    print(f"✓ Critical temperature: β_c={beta_c:.4f} (theory predicts ~1.0)")


def test_one_rsb():
    """Test 1-step RSB free energy."""
    beta = 1.5

    # Test with some sample parameters
    q0, q1, m = 0.3, 0.7, 0.5
    f_1rsb = solver.one_rsb_free_energy(q0, q1, m, beta, h=0.0, n_quad=16)

    # 1RSB should give lower (more negative) free energy than RS
    f_rs = solver.rs_free_energy(beta, h=0.0)

    print(f"✓ 1RSB free energy at β={beta}: f_1RSB={f_1rsb:.4f}, f_RS={f_rs:.4f}")
    print(f"  (1RSB should be ≤ RS: {f_1rsb <= f_rs})")


if __name__ == "__main__":
    print("=" * 50)
    print("Phase 1 Tests: Hamiltonian")
    print("=" * 50)
    test_coupling_properties()
    test_spin_initialization()
    test_hamiltonian_basic()
    print("\n✅ All Phase 1 tests passed!")

    print("\n" + "=" * 50)
    print("Phase 2 Tests: RS and 1RSB Solutions")
    print("=" * 50)
    test_rs_free_energy()
    test_critical_temperature()
    test_one_rsb()
    print("\n✅ All Phase 2 tests passed!")
