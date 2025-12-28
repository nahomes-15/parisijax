# ParisiJAX

JAX implementation of the Parisi solution for the Sherrington-Kirkpatrick spin glass model.

## Background

The Sherrington-Kirkpatrick (SK) model is a mean-field spin glass defined by the Hamiltonian

$$H = -\frac{1}{\sqrt{N}} \sum_{i<j} J_{ij} \sigma_i \sigma_j - h \sum_i \sigma_i$$

where $\sigma_i \in \\{-1, +1\\}$ are Ising spins and $J_{ij}$ are independent Gaussian random variables with $\mathbb{E}[J_{ij}] = 0$ and $\mathrm{Var}(J_{ij}) = 1/N$. The $1/\sqrt{N}$ normalization ensures an extensive free energy in the thermodynamic limit.

The model exhibits a continuous phase transition at the Almeida-Thouless (AT) line, approximately $T_c \approx 1.0$ (in units where $k_B = 1$), below which the replica-symmetric (RS) solution becomes unstable. Parisi (1980) showed that the correct low-temperature solution requires **replica symmetry breaking (RSB)**: an infinite hierarchy of ultrametric pure states characterized by a non-trivial overlap distribution $P(q)$.

### The Replica Method

The quenched free energy $f = -\beta^{-1} \mathbb{E}_J[\log Z]$ is computed via the replica trick:

$$f = -\lim_{n \to 0} \frac{\partial}{\partial n} \mathbb{E}_J[Z^n]$$

The order parameter is the **overlap matrix** $q_{\alpha\beta} = N^{-1} \sum_i \sigma_i^{(\alpha)} \sigma_i^{(\beta)}$ between replicas $\alpha, \beta \in \\{1, \ldots, n\\}$. The RS ansatz assumes $q_{\alpha\beta} = q$ for all $\alpha \neq \beta$, which yields

$$f_{\mathrm{RS}} = -\frac{\beta}{4} - \frac{1}{\beta} \int \mathcal{D}z \log 2\cosh\beta(h + z)$$

where $\mathcal{D}z = (2\pi)^{-1/2} e^{-z^2/2} dz$ is the Gaussian measure. This solution is valid for $T > T_c$ but violates the AT stability condition $\beta^2(1 - q^2) \geq 1$ at low temperatures.

### Parisi's Solution

Parisi introduced a functional order parameter $q(x)$ for $x \in [0,1]$ describing the hierarchical structure of pure states. The RSB free energy functional is given by the **Parisi PDE**:

$$\Phi_r(y) = \frac{1}{m_r} \log \int \mathcal{D}z \, \exp\left[m_r \Phi_{r+1}\left(y + \sqrt{\Delta q_r} \, z\right)\right]$$

solved backward from $\Phi_k(y) = \log 2\cosh(\beta y)$ with $\Delta q_r = q_{r+1} - q_r$ and $m_r$ the replica indices. The free energy is

$$f = -\frac{\beta(1 - q_k)}{4} - \frac{\beta}{4}\sum_r (m_{r+1} - m_r) q_r + \frac{\Phi_0(h)}{\beta}$$

In the continuous limit ($k \to \infty$), this becomes the Parisi functional with a continuous function $x(q)$, but numerical work typically uses finite $k$-RSB.

### Edwards-Anderson Order Parameter

The spin glass phase is characterized by a nonzero **Edwards-Anderson parameter**

$$q_{\mathrm{EA}} = \lim_{t \to \infty} \mathbb{E}\left[\left(\frac{1}{N}\sum_i \sigma_i(t) \sigma_i(0)\right)^2\right] = \int q^2 P(q) \, dq$$

which measures the self-overlap of the system. At high temperature ($T > T_c$), $q_{\mathrm{EA}} = 0$ and $P(q) = \delta(q)$. Below $T_c$, the distribution becomes non-trivial, reflecting the emergence of many metastable states with varying overlaps.

## Implementation

This package provides:

1. **Analytical solutions**: Replica-symmetric, 1-step RSB, and full $k$-RSB free energies via Gaussian quadrature and backward recursion.
2. **Monte Carlo sampling**: GPU-accelerated Metropolis and Gibbs samplers with automatic vectorization over disorder realizations and temperature replicas.
3. **Parallel tempering**: Enhanced equilibration for the spin glass phase via replica exchange.
4. **Overlap analysis**: Numerical computation of $P(q)$ and $q_{\mathrm{EA}}$ from equilibrium configurations.

All numerics use JAX for automatic differentiation, JIT compilation, and GPU execution.

## Installation

```bash
pip install jax jaxlib optax matplotlib numpy
git clone https://github.com/nahomes-15/parisijax
cd parisijax
pip install -e .
```

For GPU support:
```bash
pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Usage

### Basic example

```python
import jax
import jax.numpy as jnp
from parisijax.core import hamiltonian, mcmc, solver
from parisijax.analysis import overlap

# Generate disorder realization
key = jax.random.PRNGKey(42)
n_spins = 500
J = hamiltonian.sample_couplings(key, n_spins, n_samples=1)[0]

# Compute RS free energy
beta = 1.5
f_rs = solver.rs_free_energy(beta)

# Find AT critical temperature
beta_c = solver.find_critical_temperature()  # ≈ 1.1

# Run MCMC at low temperature
key, subkey = jax.random.split(key)
beta_low = 2.0
final_spins, energy_traj = mcmc.run_mcmc(
    subkey, J, beta_low, n_steps=10000, n_samples=500
)

# Sample overlap distribution
key, subkey = jax.random.split(key)
overlaps = overlap.sample_overlap_distribution(
    subkey, J, beta_low, n_samples=200, n_steps=5000, burnin=1000
)

# Compute Edwards-Anderson parameter
q_ea = overlap.compute_edwards_anderson_parameter(overlaps)
print(f"q_EA = {q_ea:.4f}")
```

### Parisi solution

```python
# Optimize k-RSB free energy
q_opt, m_opt, f_parisi, history = solver.optimize_parisi(
    beta=1.5, k=10, n_steps=1000, learning_rate=0.01
)

print(f"Parisi free energy: {f_parisi:.6f}")
print(f"Overlap breaks: {q_opt}")
```

### Parallel tempering for deep equilibration

```python
from parisijax.core.mcmc import parallel_tempering_step
from parisijax.core.hamiltonian import random_spins

# Temperature ladder
n_temps = 16
betas = jnp.geomspace(0.2, 2.0, n_temps)

# Initialize
key, subkey = jax.random.split(key)
spins_all_temps = random_spins(subkey, n_spins, n_samples=n_temps)

# Run PT
for step in range(50000):
    key, subkey = jax.random.split(key)
    spins_all_temps = parallel_tempering_step(subkey, spins_all_temps, J, betas)

# Extract samples from target temperature
coldest_samples = spins_all_temps[-1]
```

## API Reference

### `parisijax.core.hamiltonian`

- **`sample_couplings(key, n_spins, n_samples=1)`**: Generate $J_{ij} \sim \mathcal{N}(0, 1/N)$ with symmetric structure and zero diagonal.
- **`sk_energy(spins, J, h=0.0)`**: Compute $H$ for given spin configurations (vmappable).
- **`random_spins(key, n_spins, n_samples=1)`**: Sample initial configurations from uniform distribution over $\\{\pm 1\\}^N$.

### `parisijax.core.solver`

- **`rs_free_energy(beta, h=0.0, n_quad=32)`**: Replica-symmetric free energy via Gauss-Hermite quadrature.
- **`find_critical_temperature(beta_min=0.1, beta_max=2.0, tol=1e-4)`**: Locate AT instability line via binary search on $\beta^2(1 - q^2) = 1$.
- **`one_rsb_free_energy(q0, q1, m, beta, h=0.0, n_quad=32)`**: 1-step RSB solution with nested quadrature.
- **`parisi_free_energy(q_raw, m_raw, beta, h=0.0, n_quad=32, n_grid=200)`**: Full $k$-RSB via backward PDE recursion with interpolation on discretized $x$-grid.
- **`optimize_parisi(beta, h=0.0, k=10, n_steps=1000, lr=0.01)`**: Minimize $-f$ via Adam to find optimal $\\{q_r, m_r\\}$.

### `parisijax.core.mcmc`

- **`run_mcmc(key, J, beta, h=0.0, n_steps=1000, n_samples=1000, method='metropolis')`**: Vectorized MCMC with independent replicas. Returns `(final_spins, energy_trajectory)`.
- **`parallel_tempering_step(key, spins_all_temps, J, betas, h=0.0)`**: Single PT sweep with Metropolis updates and replica exchange moves.

### `parisijax.analysis.overlap`

- **`compute_overlap(spins1, spins2)`**: Compute $q = N^{-1} \sum_i \sigma_i^{(1)} \sigma_i^{(2)}$ (vmappable).
- **`sample_overlap_distribution(key, J, beta, n_samples=1000, n_steps=5000, burnin=1000)`**: Estimate $P(q)$ by running $2n$ independent chains and computing pairwise overlaps.
- **`compute_edwards_anderson_parameter(overlaps)`**: Compute $q_{\mathrm{EA}} = \langle q^2 \rangle$.
- **`theoretical_overlap_distribution(q_parisi, m_parisi)`**: Compute $P(q)$ from Parisi solution (piecewise constant with jumps at $q_r$).

## Numerical Details

### Quadrature

Gaussian expectations are evaluated using $n$-point Gauss-Hermite quadrature:

$$\int \mathcal{D}z \, f(z) \approx \sum_{i=1}^n w_i f(z_i)$$

where $\\{z_i, w_i\\}$ are nodes and weights for the standard Hermite polynomial basis. Default: $n = 32$.

### Parisi Recursion

The backward PDE is solved on a discrete grid $x \in [-10, 10]$ with linear interpolation. The recursion uses `jax.lax.scan` for efficiency and automatic differentiation through the entire computation graph.

### MCMC Equilibration

At low temperature ($\beta > 2$), simple Metropolis dynamics exhibits critical slowing down with autocorrelation time $\tau \sim \exp(c\beta)$. Parallel tempering with a geometric temperature ladder mitigates this by allowing configurations to diffuse through temperature space. Typical parameters:

- High T ($\beta \sim 0.5$): 2000 steps sufficient
- Low T ($\beta \sim 2.0$): 50,000+ PT steps recommended

## Performance

Benchmarks on Apple M1 (CPU mode):

- RS free energy: ~4 ms per temperature
- MCMC (N=500, 500 replicas, 2000 steps): 1.0 s (1M spin-flips/sec)
- Overlap sampling (200 pairs, 5000 steps): 2.5 s

Expected GPU performance (A100): 50-200M spin-flips/sec.

## References

**Theory:**

- Sherrington, D., & Kirkpatrick, S. (1975). Solvable model of a spin-glass. *Physical Review Letters*, 35(26), 1792.
- Parisi, G. (1980). A sequence of approximated solutions to the S-K model for spin glasses. *Journal of Physics A: Mathematical and General*, 13(4), L115.
- Mézard, M., Parisi, G., & Virasoro, M. A. (1987). *Spin glass theory and beyond*. World Scientific.

**Numerical methods:**

- Marinari, E., & Parisi, G. (1992). Simulated tempering: a new Monte Carlo scheme. *EPL (Europhysics Letters)*, 19(6), 451.
- Newman, M. E. J., & Barkema, G. T. (1999). *Monte Carlo methods in statistical physics*. Oxford University Press.

## Citation

```bibtex
@software{parisijax2025,
  author = {Nahom Seyoum},
  title = {ParisiJAX: JAX Implementation of Parisi Solution for Spin Glasses},
  year = {2025},
  url = {https://github.com/nahomes-15/parisijax}
}
```

## License

MIT
