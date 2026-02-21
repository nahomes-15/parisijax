"""Replica theory: RS, 1RSB, and full Parisi k-RSB free energy.

Implements the Sherrington-Kirkpatrick model solutions following
Mézard, Parisi, Virasoro "Spin Glass Theory and Beyond" (1987).
"""

from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, lax

# ============================================================================
# Result Types
# ============================================================================

class ParisiResult(NamedTuple):
    """Result of Parisi free energy optimization."""
    q: jnp.ndarray
    m: jnp.ndarray
    free_energy: float
    loss_history: jnp.ndarray
    converged: bool
    n_steps_used: int
    grad_norm_final: float


# ============================================================================
# Numerical Integration
# ============================================================================

# Cache for quadrature nodes/weights (avoid recomputation)
_QUAD_CACHE = {}

def _gauss_hermite_quadrature(n_points: int = 32) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Hermite quadrature for integral f(z) N(z|0,1) dz.

    Computes with numpy at trace time, caches raw numpy arrays to avoid
    JAX tracer leaks across JIT scopes. Returns fresh jnp arrays each call.

    Returns:
        (points, weights): z values and normalized weights summing to 1
    """
    if n_points not in _QUAD_CACHE:
        z, w = np.polynomial.hermite.hermgauss(n_points)
        z = z * np.sqrt(2.0)  # Scale to N(0,1)
        w = w / np.sqrt(np.pi)  # Normalize
        _QUAD_CACHE[n_points] = (z, w)
    z, w = _QUAD_CACHE[n_points]
    return jnp.array(z), jnp.array(w)


# ============================================================================
# Replica-Symmetric Solution
# ============================================================================

@partial(jit, static_argnums=(2, 3))
def _rs_order_parameter(
        beta: float,
        h: float = 0.0,
        n_quad: int = 32,
        max_iter: int = 100,
        tol: float = 1e-8
) -> float:
    """Solve self-consistent equation q = integral tanh^2(beta(h + sqrt(q) z)) N(z) dz.

    Args:
        beta: Inverse temperature (must be > 0)
        h: External field
        n_quad: Quadrature points (must be > 0)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        RS order parameter q in [0, 1]
    """
    z, w = _gauss_hermite_quadrature(n_quad)

    def fixed_point_iter(carry, _):
        q, converged = carry
        # q_new = integral tanh^2(beta(h + sqrt(q) z)) dmu(z)
        q_new = jnp.sum(w * jnp.tanh(beta * (h + jnp.sqrt(jnp.maximum(q, 0.0)) * z)) ** 2)
        # Damped update for stability
        q_updated = 0.5 * q + 0.5 * q_new
        new_converged = jnp.abs(q_updated - q) < tol
        return (q_updated, converged | new_converged), None

    # Initialize with high-temperature estimate
    q_init = jnp.tanh(beta) ** 2
    (q_final, _), _ = lax.scan(fixed_point_iter, (q_init, False), None, length=max_iter)

    return jnp.clip(q_final, 0.0, 1.0)


@partial(jit, static_argnums=2)
def rs_free_energy(beta: float, h: float = 0.0, n_quad: int = 32) -> float:
    """Replica-symmetric free energy: f_RS(beta, h).

    The RS free energy is:
        f_RS = -beta(1-q)^2/4 - (1/beta) integral log(2cosh(beta(h + sqrt(q) z))) N(z) dz

    where q satisfies the self-consistent equation.

    Args:
        beta: Inverse temperature
        h: External field
        n_quad: Quadrature points

    Returns:
        Free energy per spin
    """
    z, w = _gauss_hermite_quadrature(n_quad)

    # Solve for RS order parameter
    q = _rs_order_parameter(beta, h, n_quad)

    # Compute free energy
    sqrt_q = jnp.sqrt(jnp.maximum(q, 1e-10))
    log_partition = jnp.log(2.0 * jnp.cosh(beta * (h + sqrt_q * z)))
    entropic_term = -beta * (1.0 - q) ** 2 / 4.0
    field_term = -jnp.sum(w * log_partition) / beta

    return entropic_term + field_term


@partial(jit, static_argnums=3)
def find_critical_temperature(
        beta_min: float = 0.5,
        beta_max: float = 1.5,
        tol: float = 1e-6,
        n_quad: int = 32
) -> float:
    """Find AT instability: beta_c where replica symmetry breaks.

    The de Almeida-Thouless (AT) condition is:
        beta^2(1 - integral sech^4(beta sqrt(q) z) N(z) dz) = 1

    At h=0, this simplifies and beta_c = 1 exactly.

    Returns:
        Critical beta_c (theory: 1.0 for SK model at h=0)
    """
    z, w = _gauss_hermite_quadrature(n_quad)

    def at_condition(beta: float) -> float:
        # Get RS order parameter at this beta
        q = _rs_order_parameter(beta, h=0.0, n_quad=n_quad)
        sqrt_q = jnp.sqrt(jnp.maximum(q, 1e-10))

        # AT stability: lambda_AT = 1 - beta^2 integral sech^4(beta sqrt(q) z) N(z) dz
        sech4 = 1.0 / jnp.cosh(beta * sqrt_q * z) ** 4
        lambda_at = 1.0 - beta ** 2 * jnp.sum(w * sech4)
        return lambda_at

    # Binary search for where AT eigenvalue crosses zero
    def cond(carry):
        beta_low, beta_high = carry
        return beta_high - beta_low > tol

    def body(carry):
        beta_low, beta_high = carry
        beta_mid = 0.5 * (beta_low + beta_high)
        condition = at_condition(beta_mid)
        return lax.cond(
            condition > 0,  # Still stable
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
        q0, q1: Overlap parameters (0 <= q0 < q1 <= 1)
        m: RSB parameter (0 < m <= 1)
        beta: Inverse temperature
        h: External field
        n_quad: Quadrature points

    Returns:
        Free energy per spin
    """
    # Validate parameters
    q0 = jnp.clip(q0, 0.0, 1.0 - 1e-6)
    q1 = jnp.clip(q1, q0 + 1e-6, 1.0)
    m = jnp.clip(m, 1e-6, 1.0)

    z0, w0 = _gauss_hermite_quadrature(n_quad)
    z1, w1 = _gauss_hermite_quadrature(n_quad)

    sqrt_q0 = jnp.sqrt(jnp.maximum(q0, 0.0))
    sqrt_dq = jnp.sqrt(jnp.maximum(q1 - q0, 0.0))

    def inner(z0_val: float) -> float:
        arg = beta * (h + sqrt_q0 * z0_val + sqrt_dq * z1)
        log_cosh = jnp.log(2.0 * jnp.cosh(arg))
        # Numerically stable log-sum-exp
        return jax.scipy.special.logsumexp(m * log_cosh, b=w1) / m

    inner_vals = jax.vmap(inner)(z0)
    outer = jnp.sum(w0 * inner_vals)

    # Interaction energy terms (correct 1RSB formula)
    interaction = -beta * ((1.0 - q1) + m * (q1 - q0)) / 4.0

    return interaction - outer / beta


# ============================================================================
# Full Parisi k-RSB Solution
# ============================================================================

@jit
def _cumulative_softmax(x_raw: jnp.ndarray) -> jnp.ndarray:
    """Transform unconstrained -> monotone sequence in (0,1]."""
    increments = jax.nn.softplus(x_raw) + 1e-4  # Ensure positive increments
    cumsum = jnp.cumsum(increments)
    return cumsum / (cumsum[-1] + 1e-8)


@jit
def _params_to_rsb(q_raw: jnp.ndarray, m_raw: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reparametrize to valid RSB: 0 < q_0 < ... < q_k <= 1, 0 < m_0 < ... < m_k = 1."""
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
    """Parisi free energy via backward PDE recursion.

    Implements the k-RSB solution using the Cole-Hopf transform:
        Phi_r(x) = m_r^{-1} log E[exp(m_r Phi_{r+1}(x + sqrt(dq_r) z))]

    with terminal condition Phi_k(x) = log(2cosh(beta*x)).

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

    # Grid for x values (field variable)
    x_max = 5.0 + 3.0 * beta  # Adaptive range based on temperature
    x_grid = jnp.linspace(-x_max, x_max, n_grid)
    z, w = _gauss_hermite_quadrature(n_quad)

    # Compute dq increments
    q_extended = jnp.concatenate([jnp.array([0.0]), q])
    delta_q = jnp.diff(q_extended)

    # Backward recursion from r = k-1 down to r = 0
    def step(phi_next: jnp.ndarray, r: int) -> Tuple[jnp.ndarray, None]:
        # Get parameters for this level
        dq_r = delta_q[k - 1 - r]  # Reverse order
        m_r = m[k - 1 - r]

        def eval_phi(x: float) -> float:
            # Shift x by Gaussian noise scaled by sqrt(dq)
            x_shifted = x + jnp.sqrt(jnp.maximum(dq_r, 0.0)) * z
            # Interpolate phi at shifted points
            phi_interp = jnp.interp(x_shifted, x_grid, phi_next)
            # Cole-Hopf: log-sum-exp with weights
            return jax.scipy.special.logsumexp(m_r * phi_interp, b=w) / m_r

        phi_r = jax.vmap(eval_phi)(x_grid)
        return phi_r, None

    # Terminal condition: log(2*cosh(beta*x)) = |beta*x| + log(1 + exp(-2|beta*x|))
    # Using this form to avoid float32 overflow at high beta
    abs_bx = jnp.abs(beta * x_grid)
    phi_init = abs_bx + jnp.log1p(jnp.exp(-2.0 * abs_bx))
    phi_0, _ = lax.scan(step, phi_init, jnp.arange(k))

    # Evaluate at external field h
    phi_0_h = jnp.interp(h, x_grid, phi_0)

    # Interaction energy
    m_extended = jnp.concatenate([jnp.array([0.0]), m])
    delta_m = jnp.diff(m_extended)
    interaction = -beta * (1.0 - q[-1] + jnp.sum(delta_m * (q[-1] - q))) / 4.0

    return interaction - phi_0_h / beta


def optimize_parisi(
        beta: float,
        h: float = 0.0,
        k: int = 10,
        n_steps: int = 2000,
        lr: float = 0.01,
        n_quad: int = 32,
        seed: int = 0,
        tol: float = 1e-7,
        patience: int = 100,
        lr_schedule: str = 'constant',
        n_grid: int = 200,
) -> ParisiResult:
    """Optimize Parisi free energy via Adam with convergence diagnostics.

    Args:
        beta: Inverse temperature
        h: External field
        k: Number of RSB levels
        n_steps: Maximum optimization steps
        lr: Learning rate (or peak learning rate for cosine schedule)
        n_quad: Quadrature points
        seed: Random seed for initialization
        tol: Convergence tolerance on loss change
        patience: Number of steps to look back for early stopping
        lr_schedule: 'constant' or 'cosine'
        n_grid: Grid resolution for PDE discretization

    Returns:
        ParisiResult with optimized parameters and diagnostics
    """
    # Initialize with small random values
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    q_raw = jax.random.normal(key1, (k,)) * 0.1
    m_raw = jax.random.normal(key2, (k,)) * 0.1

    if lr_schedule == 'cosine':
        schedule = optax.cosine_decay_schedule(lr, n_steps)
        optimizer = optax.adam(schedule)
    else:
        optimizer = optax.adam(lr)

    opt_state = optimizer.init((q_raw, m_raw))

    @jit
    def loss_fn(params):
        # Maximize the Parisi functional (Guerra lower bound): f = sup P[x]
        return -parisi_free_energy(*params, beta, h, n_quad, n_grid)

    @jit
    def step_fn(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = jnp.sqrt(
            jnp.sum(grads[0] ** 2) + jnp.sum(grads[1] ** 2)
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, grad_norm

    # Python loop with early stopping (lax.scan doesn't support
    # dynamic early exit cleanly with optax state)
    params = (q_raw, m_raw)
    losses = []
    grad_norms = []
    converged = False
    n_steps_used = n_steps

    for i in range(n_steps):
        params, opt_state, loss, grad_norm = step_fn(params, opt_state)
        loss_val = float(loss)
        losses.append(loss_val)
        grad_norms.append(float(grad_norm))

        # Early stopping check
        if i >= patience:
            if abs(losses[i] - losses[i - patience]) < tol:
                converged = True
                n_steps_used = i + 1
                break

    q_opt, m_opt = _params_to_rsb(*params)
    f_opt = parisi_free_energy(*params, beta, h, n_quad, n_grid)

    return ParisiResult(
        q=q_opt,
        m=m_opt,
        free_energy=float(f_opt),
        loss_history=jnp.array(losses),
        converged=converged,
        n_steps_used=n_steps_used,
        grad_norm_final=grad_norms[-1] if grad_norms else float('inf'),
    )


def optimize_parisi_multistart(
        beta: float,
        h: float = 0.0,
        k: int = 10,
        n_starts: int = 5,
        **kwargs,
) -> ParisiResult:
    """Run optimize_parisi from multiple random seeds, return best result.

    Args:
        beta: Inverse temperature
        h: External field
        k: Number of RSB levels
        n_starts: Number of random initializations
        **kwargs: Additional arguments passed to optimize_parisi

    Returns:
        ParisiResult from the run with the lowest (most negative) free energy
    """
    best_result = None

    for i in range(n_starts):
        result = optimize_parisi(beta, h=h, k=k, seed=i, **kwargs)
        if best_result is None or result.free_energy < best_result.free_energy:
            best_result = result

    return best_result


# ============================================================================
# Known Results for Validation
# ============================================================================

def ground_state_energy() -> float:
    """Return the known SK ground state energy per spin.

    The Parisi formula gives E_0/N ~ -0.7633 at T=0.
    """
    return -0.7633


def high_temp_free_energy(beta: float) -> float:
    """High temperature (paramagnetic) free energy.

    For small beta: f ~ -log(2)/beta - beta/4

    Args:
        beta: Inverse temperature (must be > 0)
    """
    return -jnp.log(2.0) / beta - beta / 4.0
