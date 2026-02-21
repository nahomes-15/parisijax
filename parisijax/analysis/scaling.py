"""Finite-size scaling analysis for spin glass phase transitions.

Implements Binder cumulant, spin glass susceptibility, data collapse,
and crossing-point analysis for locating the critical temperature
of the SK model from finite-size Monte Carlo data.
"""

from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.typing import ArrayLike


class ScalingResult(NamedTuple):
    """Result of a finite-size scaling analysis."""
    beta_c: float
    nu: float
    quality: float  # residual of the data collapse


def binder_cumulant(overlaps: jnp.ndarray) -> float:
    """Binder cumulant g = (1/2)(3 - <q^4>/<q^2>^2).

    Size-independent at the critical temperature, making it ideal
    for locating T_c from the crossing of curves at different N.

    Args:
        overlaps: Array of overlap samples q_i.

    Returns:
        Binder cumulant g in [0, 1]. g -> 0 for Gaussian P(q),
        g -> 1 for delta-function P(q).
    """
    q2 = jnp.mean(overlaps ** 2)
    q4 = jnp.mean(overlaps ** 4)
    return 0.5 * (3.0 - q4 / jnp.maximum(q2 ** 2, 1e-20))


def susceptibility(overlaps: jnp.ndarray, n_spins: int) -> float:
    """Spin glass susceptibility chi_SG = N * <q^2>.

    Diverges at the critical temperature in the thermodynamic limit.

    Args:
        overlaps: Array of overlap samples.
        n_spins: System size N.

    Returns:
        Spin glass susceptibility.
    """
    return n_spins * jnp.mean(overlaps ** 2)


def correlation_length_ratio(overlaps: jnp.ndarray) -> float:
    """Correlation length ratio xi/L proxy from overlap moments.

    Uses the ratio xi/L ~ sqrt(<q^2>/<q^4> - 1) which is
    size-independent at the critical point, similar to the Binder
    cumulant but with different finite-size corrections.

    Args:
        overlaps: Array of overlap samples.

    Returns:
        Correlation length ratio proxy.
    """
    q2 = jnp.mean(overlaps ** 2)
    q4 = jnp.mean(overlaps ** 4)
    ratio = q2 ** 2 / jnp.maximum(q4, 1e-20)
    return jnp.sqrt(jnp.maximum(1.0 / jnp.maximum(ratio, 1e-20) - 1.0, 0.0))


def collect_observables(
    key: ArrayLike,
    sizes: list,
    betas: jnp.ndarray,
    n_samples: int = 200,
    n_steps: int = 2000,
    burnin: int = 500,
    h: float = 0.0,
) -> Dict[Tuple[int, float], dict]:
    """Run MCMC at all (N, beta) points and compute observables.

    For each (N, beta) pair, runs MCMC to generate overlap samples,
    then computes the Binder cumulant g, susceptibility chi, and
    Edwards-Anderson order parameter q_EA.

    Args:
        key: PRNG key.
        sizes: List of system sizes N.
        betas: Array of inverse temperatures.
        n_samples: Number of MCMC replicas per point.
        n_steps: MCMC steps per replica.
        burnin: Burn-in steps to discard.
        h: External field.

    Returns:
        Dict keyed by (N, beta) with values containing
        'g' (Binder), 'chi' (susceptibility), 'q_EA' (EA parameter),
        'overlaps' (raw overlap samples).
    """
    from parisijax.analysis.overlap import compute_overlap
    from parisijax.core.hamiltonian import sample_couplings
    from parisijax.core.mcmc import run_mcmc

    results = {}
    betas_arr = np.asarray(betas)

    for n_spins in sizes:
        # Generate one disorder realization per size
        key, subkey = random.split(key)
        J = sample_couplings(subkey, n_spins, 1)[0]

        for beta_val in betas_arr:
            beta_val = float(beta_val)
            key, subkey = random.split(key)

            # Run MCMC with 2*n_samples chains for pairing
            final_spins, _ = run_mcmc(
                subkey, J, beta_val, h=h,
                n_steps=n_steps + burnin,
                n_samples=2 * n_samples,
                method='metropolis',
            )

            # Compute pairwise overlaps
            spins1, spins2 = final_spins[::2], final_spins[1::2]
            overlaps = jax.vmap(compute_overlap)(spins1, spins2)

            g = float(binder_cumulant(overlaps))
            chi = float(susceptibility(overlaps, n_spins))
            q_EA = float(jnp.mean(overlaps ** 2))

            results[(n_spins, beta_val)] = {
                'g': g,
                'chi': chi,
                'q_EA': q_EA,
                'overlaps': overlaps,
            }

    return results


def data_collapse(
    observables: Dict[Tuple[int, float], dict],
    sizes: list,
    betas: jnp.ndarray,
    observable_key: str = 'g',
    beta_c_init: float = 1.0,
    nu_init: float = 2.5,
    n_restarts: int = 5,
) -> ScalingResult:
    """Minimize the data-collapse residual for N^(1/nu)(beta - beta_c) scaling.

    For the SK model (mean-field), the finite-size scaling form is:
        O(N, beta) = f(N^(2/5) (beta - beta_c))
    where 2/5 = 1/(d*nu) with d=infinity giving nu=5/2 for mean-field.

    Args:
        observables: Dict from collect_observables.
        sizes: List of system sizes.
        betas: Array of inverse temperatures.
        observable_key: Which observable to collapse ('g', 'chi', 'q_EA').
        beta_c_init: Initial guess for critical temperature.
        nu_init: Initial guess for correlation length exponent.
        n_restarts: Number of random restarts for optimization.

    Returns:
        ScalingResult with (beta_c, nu, quality).
    """
    from scipy.optimize import minimize

    betas_arr = np.asarray(betas)

    def collapse_residual(params):
        beta_c, nu = params
        if nu <= 0:
            return 1e10

        # Compute scaled variable x = N^(1/(d*nu)) * (beta - beta_c)
        # For mean-field (infinite d), use N^(2/5) as the scaling power
        # More generally, use N^(1/(3*nu)) as a proxy
        all_x = []
        all_y = []

        for n_spins in sizes:
            scaling_power = 1.0 / (3.0 * nu)
            for beta_val in betas_arr:
                beta_val = float(beta_val)
                key = (n_spins, beta_val)
                if key not in observables:
                    continue
                x_scaled = (n_spins ** scaling_power) * (beta_val - beta_c)
                y_val = observables[key][observable_key]
                all_x.append(x_scaled)
                all_y.append(y_val)

        if len(all_x) < 3:
            return 1e10

        all_x = np.array(all_x)
        all_y = np.array(all_y)

        # Sort by x for interpolation-based quality metric
        order = np.argsort(all_x)
        all_x = all_x[order]
        all_y = all_y[order]

        # Residual: sum of squared differences between nearby points
        # from different sizes (they should overlap after scaling)
        residual = 0.0
        count = 0
        for i in range(len(all_x) - 1):
            dx = all_x[i + 1] - all_x[i]
            if abs(dx) < 1e-10:
                continue
            dy = all_y[i + 1] - all_y[i]
            residual += (dy / (1.0 + abs(dx))) ** 2
            count += 1

        return residual / max(count, 1)

    best_result = None
    best_quality = np.inf

    rng = np.random.RandomState(42)
    for i in range(n_restarts):
        if i == 0:
            x0 = [beta_c_init, nu_init]
        else:
            x0 = [beta_c_init + rng.randn() * 0.1, nu_init + rng.randn() * 0.5]

        result = minimize(
            collapse_residual, x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6},
        )

        if result.fun < best_quality:
            best_quality = result.fun
            best_result = result

    beta_c, nu = best_result.x
    return ScalingResult(beta_c=float(beta_c), nu=float(nu), quality=float(best_quality))


def find_binder_crossing(
    observables: Dict[Tuple[int, float], dict],
    sizes: list,
    betas: jnp.ndarray,
) -> float:
    """Find the crossing point of Binder cumulant curves for two largest sizes.

    Interpolates Binder cumulant vs beta for the two largest sizes
    and finds where the curves intersect.

    Args:
        observables: Dict from collect_observables.
        sizes: List of system sizes (at least 2).
        betas: Array of inverse temperatures.

    Returns:
        beta_c estimate from the crossing point.
    """
    sorted_sizes = sorted(sizes)
    n1, n2 = sorted_sizes[-2], sorted_sizes[-1]
    betas_arr = np.asarray(betas, dtype=np.float64)

    g1 = np.array([observables[(n1, float(b))]['g'] for b in betas_arr])
    g2 = np.array([observables[(n2, float(b))]['g'] for b in betas_arr])

    # Find where g2 - g1 changes sign (crossing)
    diff = g2 - g1
    crossings = np.where(np.diff(np.sign(diff)))[0]

    if len(crossings) == 0:
        # No crossing found; return midpoint as fallback
        return float(np.mean(betas_arr))

    # Linear interpolation at the first crossing
    i = crossings[0]
    b_lo, b_hi = float(betas_arr[i]), float(betas_arr[i + 1])
    d_lo, d_hi = float(diff[i]), float(diff[i + 1])

    if abs(d_hi - d_lo) < 1e-20:
        return 0.5 * (b_lo + b_hi)

    beta_c = b_lo - d_lo * (b_hi - b_lo) / (d_hi - d_lo)
    return float(beta_c)
