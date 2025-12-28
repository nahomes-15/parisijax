"""Replica theory: RS, 1RSB, and full Parisi k-RSB free energy."""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, lax
from jax.typing import ArrayLike


# ============================================================================
# Numerical Integration
# ============================================================================

@partial(jit, static_argnums=0)
def _gauss_hermite_quadrature(n_points: int = 32) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Hermite quadrature for ∫ f(z) 𝒩(z|0,1) dz.

    Returns:
        (points, weights): z values and normalized weights summing to 1
    """
    z, w = np.polynomial.hermite.hermgauss(n_points)
    z = jnp.array(z) * jnp.sqrt(2.0)  # Scale to N(0,1)
    w = jnp.array(w) / jnp.sqrt(jnp.pi)  # Normalize
    return z, w


# ============================================================================
# Replica-Symmetric Solution
# ============================================================================

@partial(jit, static_argnums=2)
def rs_free_energy(beta: float, h: float = 0.0, n_quad: int = 32) -> float:
    """Replica-symmetric free energy: f = -β/4 - β⁻¹ ∫ log 2cosh(β(h+z)) 𝒩(z|0,1) dz

    Args:
        beta: Inverse temperature
        h: External field
        n_quad: Quadrature points

    Returns:
        Free energy per spin
    """
    z, w = _gauss_hermite_quadrature(n_quad)
    log_partition = jnp.log(2.0 * jnp.cosh(beta * (h + z)))
    return -beta / 4.0 - jnp.sum(w * log_partition) / beta


@partial(jit, static_argnums=3)
def find_critical_temperature(
    beta_min: float = 0.1,
    beta_max: float = 2.0,
    tol: float = 1e-4,
    n_quad: int = 32
) -> float:
    """Find AT instability via binary search on β²(1 - q²) = 1.

    Returns:
        Critical βc (theory: ~1.0 for SK model)
    """
    z, w = _gauss_hermite_quadrature(n_quad)

    def at_condition(beta: float) -> float:
        q = jnp.sum(w * jnp.tanh(beta * z) ** 2)
        return beta ** 2 * (1.0 - q ** 2) - 1.0

    # Binary search
    def cond(carry):
        beta_low, beta_high = carry
        return beta_high - beta_low > tol

    def body(carry):
        beta_low, beta_high = carry
        beta_mid = 0.5 * (beta_low + beta_high)
        condition = at_condition(beta_mid)
        return lax.cond(
            condition < 0,
            lambda _: (beta_mid, beta_high),
            lambda _: (beta_low, beta_mid),
            None
        )

    beta_low, beta_high = lax.while_loop(cond, body, (beta_min, beta_max))
    return 0.5 * (beta_low + beta_high)


# ============================================================================
# 1-Step RSB Solution
# ============================================================================

@partial(jit, static_argnums=5)
def one_rsb_free_energy(
    q0: float,
    q1: float,
    m: float,
    beta: float,
    h: float = 0.0,
    n_quad: int = 32
) -> float:
    """1RSB free energy with nested Gaussian integrals.

    Args:
        q0, q1: Overlap parameters (0 < q0 < q1 ≤ 1)
        m: RSB parameter (0 < m ≤ 1)
        beta: Inverse temperature
        h: External field
        n_quad: Quadrature points

    Returns:
        Free energy per spin
    """
    z0, w0 = _gauss_hermite_quadrature(n_quad)
    z1, w1 = _gauss_hermite_quadrature(n_quad)

    def inner(z0_val: float) -> float:
        arg = beta * (h + jnp.sqrt(q0) * z0_val + jnp.sqrt(q1 - q0) * z1)
        log_terms = jnp.log(2.0 * jnp.cosh(arg))
        return jnp.log(jnp.sum(w1 * jnp.exp(m * log_terms))) / m

    inner_vals = jax.vmap(inner)(z0)
    outer = jnp.sum(w0 * inner_vals)

    return -beta * (1.0 - q1) / 4.0 - beta * m * (q1 - q0) / 4.0 + outer / beta


# ============================================================================
# Full Parisi k-RSB Solution
# ============================================================================

@jit
def _cumulative_softmax(x_raw: jnp.ndarray) -> jnp.ndarray:
    """Transform unconstrained → monotone sequence in (0,1]."""
    increments = jax.nn.softplus(x_raw)
    cumsum = jnp.cumsum(increments)
    return cumsum / (cumsum[-1] + 1e-8)


@jit
def _params_to_rsb(q_raw: jnp.ndarray, m_raw: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reparametrize to valid RSB: 0 < q₀ < ... < qₖ ≤ 1, 0 < m₀ < ... < mₖ = 1."""
    q = _cumulative_softmax(q_raw)
    m = jnp.cumsum(jax.nn.softmax(m_raw))
    return q, m


@partial(jit, static_argnums=(4, 5))
def parisi_free_energy(
    q_raw: jnp.ndarray,
    m_raw: jnp.ndarray,
    beta: float,
    h: float = 0.0,
    n_quad: int = 32,
    n_grid: int = 200
) -> float:
    """Parisi free energy via backward PDE recursion: Φᵣ(x) = mᵣ⁻¹ log 𝔼[exp(mᵣ Φᵣ₊₁(x + √Δqᵣ z))].

    Args:
        q_raw, m_raw: Unconstrained parameters (length k)
        beta: Inverse temperature
        h: External field
        n_quad: Quadrature points for Gaussian integrals
        n_grid: Grid resolution for x discretization

    Returns:
        Free energy per spin
    """
    q, m = _params_to_rsb(q_raw, m_raw)
    k = len(q)

    x_grid = jnp.linspace(-10.0, 10.0, n_grid)
    z, w = _gauss_hermite_quadrature(n_quad)

    # Compute Δq
    delta_q = jnp.diff(jnp.concatenate([jnp.array([0.0]), q]))

    # Backward recursion
    def step(phi_next: jnp.ndarray, r: int) -> Tuple[jnp.ndarray, None]:
        dq_r = delta_q[r + 1]
        m_r = m[r]

        def eval_phi(x: float) -> float:
            x_shifted = x + jnp.sqrt(dq_r) * z
            phi_interp = jnp.interp(x_shifted, x_grid, phi_next)
            return jax.scipy.special.logsumexp(m_r * phi_interp, b=w) / m_r

        phi_r = jax.vmap(eval_phi)(x_grid)
        return phi_r, None

    phi_init = jnp.log(2.0 * jnp.cosh(beta * x_grid))
    phi_0, _ = lax.scan(step, phi_init, jnp.arange(k - 1, -1, -1))

    # Free energy components
    phi_0_h = jnp.interp(h, x_grid, phi_0)
    entropy = -beta * (1.0 - q[-1]) / 4.0
    interaction = -beta * jnp.sum(jnp.diff(jnp.concatenate([jnp.array([0.0]), m])) * q) / 4.0

    return entropy + interaction + phi_0_h / beta


def optimize_parisi(
    beta: float,
    h: float = 0.0,
    k: int = 10,
    n_steps: int = 1000,
    lr: float = 0.01,
    n_quad: int = 32
) -> Tuple[jnp.ndarray, jnp.ndarray, float, list]:
    """Optimize Parisi free energy via Adam.

    Args:
        beta: Inverse temperature
        h: External field
        k: Number of RSB levels
        n_steps: Optimization steps
        lr: Learning rate
        n_quad: Quadrature points

    Returns:
        (q_opt, m_opt, f_opt, loss_history)
    """
    # Initialize
    key = jax.random.PRNGKey(0)
    q_raw = jax.random.normal(key, (k,)) * 0.1
    m_raw = jax.random.normal(jax.random.PRNGKey(1), (k,)) * 0.1

    optimizer = optax.adam(lr)
    opt_state = optimizer.init((q_raw, m_raw))

    @jit
    def loss_fn(params):
        return -parisi_free_energy(*params, beta, h, n_quad)

    @jit
    def step_fn(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Optimize
    params = (q_raw, m_raw)
    history = []
    for _ in range(n_steps):
        params, opt_state, loss = step_fn(params, opt_state)
        history.append(float(loss))

    q_opt, m_opt = _params_to_rsb(*params)
    f_opt = parisi_free_energy(*params, beta, h, n_quad)

    return q_opt, m_opt, f_opt, history
