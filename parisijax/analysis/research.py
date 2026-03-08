"""Research module: k-RSB convergence, autodiff thermodynamics, and AT line.

Leverages JAX automatic differentiation through the Parisi backward PDE
to compute thermodynamic quantities and study convergence properties
that are inaccessible to conventional numerical approaches.
"""

from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from parisijax.core.solver import (
    _params_to_rsb,
    optimize_parisi,
    parisi_free_energy,
    rs_free_energy,
    _rs_order_parameter,
    _gauss_hermite_quadrature,
)


# ============================================================================
# Result Types
# ============================================================================

class ConvergenceResult(NamedTuple):
    """Result of k-RSB convergence analysis."""
    k_values: np.ndarray
    free_energies: np.ndarray
    delta_f: np.ndarray       # f_k - f_{k-1}
    q_ea_values: np.ndarray   # Edwards-Anderson parameter at each k
    converged_flags: np.ndarray
    beta: float
    h: float


class ThermodynamicResult(NamedTuple):
    """Thermodynamic quantities from autodiff."""
    beta: float
    h: float
    free_energy: float
    internal_energy: float
    entropy: float
    specific_heat: float
    magnetization: float
    susceptibility: float


class ATLineResult(NamedTuple):
    """Result of AT line computation."""
    h_values: np.ndarray
    beta_c_values: np.ndarray
    t_c_values: np.ndarray


# ============================================================================
# k-RSB Convergence Analysis
# ============================================================================

def krsb_convergence(
    beta: float,
    h: float = 0.0,
    k_values: list = None,
    n_steps: int = 3000,
    lr: float = 0.01,
    n_quad: int = 48,
    n_grid: int = 300,
    n_starts: int = 3,
) -> ConvergenceResult:
    """Study convergence of k-RSB free energy as k -> infinity.

    Computes the optimized Parisi free energy f_k for each k in k_values
    and analyzes the convergence rate.

    Args:
        beta: Inverse temperature (should be > 1 for RSB regime).
        h: External field.
        k_values: List of RSB levels to compute. Default: [1,2,...,20].
        n_steps: Optimization steps per run.
        lr: Learning rate.
        n_quad: Quadrature points.
        n_grid: PDE grid resolution.
        n_starts: Multi-start seeds per k.

    Returns:
        ConvergenceResult with free energies and convergence diagnostics.
    """
    if k_values is None:
        k_values = list(range(1, 21))

    free_energies = []
    q_ea_values = []
    converged_flags = []

    for k in k_values:
        best_f = np.inf
        best_q_ea = 0.0
        best_converged = False

        for seed in range(n_starts):
            result = optimize_parisi(
                beta, h=h, k=k,
                n_steps=n_steps, lr=lr,
                n_quad=n_quad, n_grid=n_grid,
                seed=seed, lr_schedule='cosine',
            )
            if result.free_energy < best_f:
                best_f = result.free_energy
                best_q_ea = float(result.q[-1])
                best_converged = result.converged

        free_energies.append(best_f)
        q_ea_values.append(best_q_ea)
        converged_flags.append(best_converged)

    free_energies = np.array(free_energies)
    delta_f = np.diff(free_energies, prepend=0.0)

    return ConvergenceResult(
        k_values=np.array(k_values),
        free_energies=free_energies,
        delta_f=delta_f,
        q_ea_values=np.array(q_ea_values),
        converged_flags=np.array(converged_flags),
        beta=beta,
        h=h,
    )


def fit_convergence_rate(result: ConvergenceResult, k_min: int = 3):
    """Fit the convergence rate: f_k - f_inf ~ C / k^alpha.

    Uses the last computed value as a proxy for f_inf and fits
    the power law to the tail of the sequence.

    Args:
        result: Output of krsb_convergence.
        k_min: Minimum k to include in the fit (skip initial transient).

    Returns:
        (alpha, C, f_inf_estimate): Power law exponent, prefactor,
        and estimated limiting free energy.
    """
    k = result.k_values
    f = result.free_energies

    # Use Richardson extrapolation estimate for f_inf
    # f_inf ~ (k2^a * f(k2) - k1^a * f(k1)) / (k2^a - k1^a)
    # Start with simple estimate: f_inf = f[-1]
    f_inf = f[-1]

    mask = k >= k_min
    k_fit = k[mask].astype(float)
    residuals = np.abs(f[mask] - f_inf)

    # Avoid log of zero
    valid = residuals > 1e-15
    if np.sum(valid) < 2:
        return 1.0, 0.0, f_inf

    log_k = np.log(k_fit[valid])
    log_r = np.log(residuals[valid])

    # Linear fit in log-log space
    coeffs = np.polyfit(log_k, log_r, 1)
    alpha = -coeffs[0]
    C = np.exp(coeffs[1])

    return float(alpha), float(C), float(f_inf)


# ============================================================================
# Autodiff Thermodynamics
# ============================================================================

def _make_free_energy_fn(k: int, n_quad: int = 48, n_grid: int = 300,
                         n_steps: int = 3000, n_starts: int = 3):
    """Create a function that computes the optimized free energy f(beta, h).

    Returns a pure function suitable for JAX differentiation.
    """
    def f_opt(beta, h):
        best_f = np.inf
        for seed in range(n_starts):
            result = optimize_parisi(
                float(beta), h=float(h), k=k,
                n_steps=n_steps, lr=0.01,
                n_quad=n_quad, n_grid=n_grid,
                seed=seed, lr_schedule='cosine',
            )
            if result.free_energy < best_f:
                best_f = result.free_energy
        return best_f
    return f_opt


def thermodynamic_quantities(
    beta: float,
    h: float = 0.0,
    k: int = 15,
    n_quad: int = 48,
    n_grid: int = 300,
    n_steps: int = 3000,
    dbeta: float = 1e-3,
    dh: float = 1e-3,
) -> ThermodynamicResult:
    """Compute thermodynamic quantities via numerical differentiation.

    Uses central finite differences of the optimized Parisi free energy
    to extract physical observables.

    The key relations:
        f = free energy per spin
        E = d(beta*f)/d(beta)             (internal energy)
        S = beta*(E - f) = beta^2 df/dbeta  (entropy ... S = beta(E-f))
        C = -beta^2 dE/dbeta               (specific heat)
        M = -df/dh                          (magnetization)
        chi = -d^2f/dh^2                    (susceptibility)

    Args:
        beta: Inverse temperature.
        h: External field.
        k: RSB levels for the solver.
        n_quad: Quadrature points.
        n_grid: PDE grid points.
        n_steps: Optimization steps.
        dbeta: Step size for beta derivatives.
        dh: Step size for h derivatives.

    Returns:
        ThermodynamicResult with all quantities.
    """
    f_fn = _make_free_energy_fn(k, n_quad, n_grid, n_steps)

    # Central differences for beta derivatives
    f0 = f_fn(beta, h)
    f_bp = f_fn(beta + dbeta, h)
    f_bm = f_fn(beta - dbeta, h)

    # d(beta*f)/dbeta via central difference
    bf_bp = (beta + dbeta) * f_bp
    bf_bm = (beta - dbeta) * f_bm
    internal_energy = (bf_bp - bf_bm) / (2 * dbeta)

    # Entropy: S = beta * (E - f)
    entropy = beta * (internal_energy - f0)

    # Specific heat: C = -beta^2 * dE/dbeta
    # Need E at beta +/- dbeta
    bf0 = beta * f0
    e_bp = bf_bp  # E(beta+dbeta) ~ d(beta*f)/dbeta at beta+dbeta...
    # Actually, let's use the simpler formula:
    # C = beta^2 * d^2(beta*f)/dbeta^2
    # = beta^2 * (bf_bp - 2*bf0 + bf_bm) / dbeta^2
    specific_heat = beta**2 * (bf_bp - 2 * bf0 + bf_bm) / dbeta**2

    # h derivatives for magnetization and susceptibility
    f_hp = f_fn(beta, h + dh)
    f_hm = f_fn(beta, h - dh)

    magnetization = -(f_hp - f_hm) / (2 * dh)
    susceptibility = -(f_hp - 2 * f0 + f_hm) / dh**2

    return ThermodynamicResult(
        beta=beta,
        h=h,
        free_energy=f0,
        internal_energy=internal_energy,
        entropy=entropy,
        specific_heat=specific_heat,
        magnetization=magnetization,
        susceptibility=susceptibility,
    )


def thermodynamic_sweep(
    betas: np.ndarray,
    h: float = 0.0,
    k: int = 15,
    n_quad: int = 48,
    n_grid: int = 300,
    n_steps: int = 3000,
    dbeta: float = 1e-3,
    dh: float = 1e-3,
) -> dict:
    """Compute thermodynamic quantities across a range of temperatures.

    Args:
        betas: Array of inverse temperatures.
        h: External field.
        k: RSB levels.
        Other args: passed to thermodynamic_quantities.

    Returns:
        Dict with arrays: 'beta', 'free_energy', 'internal_energy',
        'entropy', 'specific_heat', 'magnetization', 'susceptibility'.
    """
    results = {
        'beta': [], 'free_energy': [], 'internal_energy': [],
        'entropy': [], 'specific_heat': [], 'magnetization': [],
        'susceptibility': [],
    }

    for beta in betas:
        beta = float(beta)
        thermo = thermodynamic_quantities(
            beta, h=h, k=k, n_quad=n_quad, n_grid=n_grid,
            n_steps=n_steps, dbeta=dbeta, dh=dh,
        )
        results['beta'].append(thermo.beta)
        results['free_energy'].append(thermo.free_energy)
        results['internal_energy'].append(thermo.internal_energy)
        results['entropy'].append(thermo.entropy)
        results['specific_heat'].append(thermo.specific_heat)
        results['magnetization'].append(thermo.magnetization)
        results['susceptibility'].append(thermo.susceptibility)

    return {k: np.array(v) for k, v in results.items()}


# ============================================================================
# de Almeida-Thouless Line
# ============================================================================

@partial(jit, static_argnums=(2,))
def _at_eigenvalue(beta: float, h: float, n_quad: int = 48) -> float:
    """Compute the AT stability eigenvalue lambda_AT(beta, h).

    The RS solution is unstable (RSB occurs) when lambda_AT < 0.

    lambda_AT = 1 - beta^2 * integral sech^4(beta(h + sqrt(q)*z)) dmu(z)

    where q is the RS order parameter at (beta, h).

    Args:
        beta: Inverse temperature.
        h: External field.
        n_quad: Quadrature points.

    Returns:
        AT eigenvalue. Negative means RS is unstable.
    """
    z, w = _gauss_hermite_quadrature(n_quad)
    q = _rs_order_parameter(beta, h, n_quad)
    sqrt_q = jnp.sqrt(jnp.maximum(q, 1e-10))

    sech4 = 1.0 / jnp.cosh(beta * (h + sqrt_q * z)) ** 4
    return 1.0 - beta ** 2 * jnp.sum(w * sech4)


def compute_at_line(
    h_values: np.ndarray = None,
    beta_range: Tuple[float, float] = (0.5, 5.0),
    tol: float = 1e-6,
    n_quad: int = 48,
) -> ATLineResult:
    """Compute the de Almeida-Thouless line in the (h, T) plane.

    For each h, finds beta_c(h) where the AT eigenvalue crosses zero,
    marking the boundary between RS stability and RSB.

    The AT line satisfies:
        1 = beta^2 integral sech^4(beta(h + sqrt(q)*z)) dmu(z)

    At h=0: beta_c = 1 (T_c = 1) exactly.
    As h -> inf: beta_c -> inf (T_c -> 0), RS is always stable.

    Args:
        h_values: External field values to trace. Default: 0 to 1.5.
        beta_range: Search range for beta_c.
        tol: Bisection tolerance.
        n_quad: Quadrature points.

    Returns:
        ATLineResult with h, beta_c, and T_c arrays.
    """
    if h_values is None:
        h_values = np.linspace(0.0, 1.5, 30)

    beta_c_values = []
    beta_min_default, beta_max_default = beta_range

    for h_val in h_values:
        h_val = float(h_val)

        # Adaptive search range
        beta_lo = beta_min_default
        beta_hi = beta_max_default

        # Ensure sign change exists: AT eigenvalue should be positive at
        # low beta (RS stable) and negative at high beta (RS unstable)
        lam_lo = float(_at_eigenvalue(beta_lo, h_val, n_quad))
        lam_hi = float(_at_eigenvalue(beta_hi, h_val, n_quad))

        # If no instability found in range, extend
        while lam_hi > 0 and beta_hi < 50.0:
            beta_hi *= 2.0
            lam_hi = float(_at_eigenvalue(beta_hi, h_val, n_quad))

        if lam_hi > 0:
            # RS is stable everywhere in range (large h regime)
            beta_c_values.append(np.inf)
            continue

        # Bisection
        for _ in range(100):
            if beta_hi - beta_lo < tol:
                break
            beta_mid = 0.5 * (beta_lo + beta_hi)
            lam_mid = float(_at_eigenvalue(beta_mid, h_val, n_quad))
            if lam_mid > 0:
                beta_lo = beta_mid
            else:
                beta_hi = beta_mid

        beta_c_values.append(0.5 * (beta_lo + beta_hi))

    beta_c_arr = np.array(beta_c_values)
    t_c_arr = np.where(np.isfinite(beta_c_arr), 1.0 / beta_c_arr, 0.0)

    return ATLineResult(
        h_values=h_values,
        beta_c_values=beta_c_arr,
        t_c_values=t_c_arr,
    )


# ============================================================================
# RS vs RSB Free Energy Comparison
# ============================================================================

def rs_vs_rsb_comparison(
    betas: np.ndarray,
    h: float = 0.0,
    k: int = 15,
    n_quad: int = 48,
    n_grid: int = 300,
    n_steps: int = 3000,
) -> dict:
    """Compare RS and optimized RSB free energies across temperatures.

    The difference f_RS - f_RSB quantifies the "cost of replica symmetry":
    how much free energy is missed by the simpler RS approximation.

    Args:
        betas: Array of inverse temperatures.
        h: External field.
        k: RSB levels for Parisi solver.

    Returns:
        Dict with 'beta', 'f_rs', 'f_rsb', 'delta_f' arrays.
    """
    f_rs_list = []
    f_rsb_list = []

    for beta in betas:
        beta = float(beta)

        # RS free energy
        f_rs = float(rs_free_energy(beta, h, n_quad))
        f_rs_list.append(f_rs)

        # Parisi RSB free energy
        best_f = np.inf
        for seed in range(3):
            result = optimize_parisi(
                beta, h=h, k=k,
                n_steps=n_steps, lr=0.01,
                n_quad=n_quad, n_grid=n_grid,
                seed=seed, lr_schedule='cosine',
            )
            if result.free_energy < best_f:
                best_f = result.free_energy
        f_rsb_list.append(best_f)

    f_rs = np.array(f_rs_list)
    f_rsb = np.array(f_rsb_list)

    return {
        'beta': np.array(betas),
        'f_rs': f_rs,
        'f_rsb': f_rsb,
        'delta_f': f_rs - f_rsb,
    }


# ============================================================================
# Parisi Order Parameter Function x(q)
# ============================================================================

def extract_parisi_function(
    beta: float,
    h: float = 0.0,
    k: int = 20,
    n_quad: int = 48,
    n_grid: int = 300,
    n_steps: int = 4000,
    n_starts: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the Parisi order parameter function x(q).

    The k-RSB solution approximates x(q) as a step function with k levels.
    As k -> inf, this converges to the continuous x(q) of full RSB.

    Args:
        beta: Inverse temperature.
        h: External field.
        k: Number of RSB levels (higher = finer approximation).

    Returns:
        (q_values, x_values): The step function representation of x(q).
        q_values has length k+1 (including q=0 and q=q_EA).
        x_values has length k (the step heights).
    """
    best_result = None
    best_f = np.inf

    for seed in range(n_starts):
        result = optimize_parisi(
            beta, h=h, k=k,
            n_steps=n_steps, lr=0.01,
            n_quad=n_quad, n_grid=n_grid,
            seed=seed, lr_schedule='cosine',
        )
        if result.free_energy < best_f:
            best_f = result.free_energy
            best_result = result

    q = np.array(best_result.q)
    m = np.array(best_result.m)

    # Build step function: x(q) = m_r for q_{r-1} < q < q_r
    q_values = np.concatenate([[0.0], q])
    x_values = m

    return q_values, x_values
