# ParisiJax

**Differentiable Spin Glass Solvers in JAX**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-Powered-blue.svg)](https://github.com/google/jax)

**ParisiJax** is a high-performance, fully differentiable library for solving the **Sherrington-Kirkpatrick (SK)** model and other mean-field spin glasses. It connects the infinite-*N* theoretical limit (Parisi Formula) to finite-*N* reality (Monte Carlo simulations).

---

## Highlights

**Ground state energy in 3 lines:**

```python
from parisijax.core.solver import optimize_parisi
result = optimize_parisi(beta=10.0, k=20, n_steps=3000)
print(f"E_0/N = {result.free_energy:.4f}")  # ≈ -0.7633
```

**1000-replica GPU MCMC in 5 lines:**

```python
from parisijax.core.hamiltonian import sample_couplings
from parisijax.core.mcmc import run_mcmc
import jax

J = sample_couplings(jax.random.PRNGKey(0), 1024, 1)[0]
spins, energies = run_mcmc(jax.random.PRNGKey(1), J, beta=2.0, n_samples=1000, n_steps=5000)
```

**Differentiate through the Parisi PDE:**

```python
import jax
from parisijax.core.solver import parisi_free_energy

grad_fn = jax.grad(parisi_free_energy, argnums=(0, 1))
dq, dm = grad_fn(q_raw, m_raw, beta=1.5)  # Gradients flow through the full k-RSB recursion
```

---

## Validated Results

| Quantity | Known Value | ParisiJax | Source |
|----------|-------------|-----------|--------|
| Ground state energy *E*₀/*N* | -0.7633 | -0.763x | Parisi (1980) |
| Critical temperature *β*_c | 1.0 | ~1.04 | de Almeida & Thouless (1978) |
| *q* = 0 above *T*_c | 0 | < 0.05 | RS theory |
| *q* > 0 below *T*_c | > 0 | > 0.1 | RSB theory |

---

## Installation

```bash
pip install parisijax
```

For development:

```bash
git clone https://github.com/nahomes-15/parisijax.git
cd parisijax
pip install -e ".[dev]"
```

For GPU support (CUDA 12):

```bash
pip install parisijax[gpu]
```

---

## Quick Start

```python
import jax
import jax.numpy as jnp
from parisijax.core.hamiltonian import sample_couplings
from parisijax.core.solver import rs_free_energy, optimize_parisi
from parisijax.core.mcmc import run_mcmc
from parisijax.analysis.overlap import compute_overlap

# 1. Generate an SK instance
key = jax.random.PRNGKey(42)
N = 256
J = sample_couplings(key, N, 1)[0]

# 2. Compute RS free energy across temperatures
betas = jnp.linspace(0.3, 2.5, 20)
f_rs = [float(rs_free_energy(b)) for b in betas]

# 3. Optimize the full Parisi solution at low temperature
result = optimize_parisi(beta=2.0, k=10, n_steps=1500)
print(f"Parisi free energy: {result.free_energy:.4f}")
print(f"Converged: {result.converged} in {result.n_steps_used} steps")

# 4. Run MCMC and measure overlaps
key1, key2 = jax.random.split(key)
spins, energies = run_mcmc(key1, J, beta=2.0, n_samples=200, n_steps=3000)
q = jax.vmap(compute_overlap, in_axes=(0, 0))(spins[::2], spins[1::2])
print(f"Mean |q|: {float(jnp.mean(jnp.abs(q))):.3f}")
```

---

## The Physics

### The SK Hamiltonian

The Sherrington-Kirkpatrick model describes *N* Ising spins with quenched Gaussian disorder:

$$H_N(\boldsymbol{\sigma}) = -\frac{1}{\sqrt{N}} \sum_{i < j} J_{ij} \sigma_i \sigma_j - h \sum_i \sigma_i$$

where $J_{ij} \sim \mathcal{N}(0, 1)$ are i.i.d. random couplings and *h* is an external field.

### The Parisi Variational Formula

In the thermodynamic limit, the quenched free energy density converges to $f(\beta, h) = \inf_{x \in \mathcal{X}} \mathcal{P}[x]$, where the functional is defined by the **Parisi backward PDE**:

$$\frac{\partial \Phi}{\partial q} + \frac{1}{2} \frac{\partial^2 \Phi}{\partial y^2} + \frac{1}{2} x(q) \left( \frac{\partial \Phi}{\partial y} \right)^2 = 0$$

with terminal condition $\Phi(1, y) = \log \cosh(\beta y)$.

### k-RSB Approximation

ParisiJax discretizes $x(q)$ as a step function with *k* levels. The PDE becomes a recursive sequence of Gaussian convolutions:

$$f_i(y) = \frac{1}{m_i} \log \int \mathcal{D}z \, \exp\left( m_i f_{i+1}\left( y + z\sqrt{q_{i+1} - q_i} \right) \right)$$

implemented via `jax.lax.scan` and differentiable Gauss-Hermite quadrature, enabling gradient-based optimization of the variational parameters $\{q_i, m_i\}$.

### Finite-Size Scaling

The `analysis.scaling` module provides tools for extracting critical exponents from Monte Carlo data:
- **Binder cumulant** $g = \frac{1}{2}(3 - \langle q^4 \rangle / \langle q^2 \rangle^2)$ — size-independent at *T*_c
- **Spin glass susceptibility** $\chi_{SG} = N \langle q^2 \rangle$ — diverges at *T*_c
- **Data collapse** — minimize residual for $N^{1/(d\nu)}(\beta - \beta_c)$ scaling

---

## API Reference

### `parisijax.core.hamiltonian`

| Function | Description |
|----------|-------------|
| `sample_couplings(key, n_spins, n_samples)` | Generate symmetric SK coupling matrices |
| `sk_energy(spins, J, h)` | Compute SK Hamiltonian energy |
| `random_spins(key, n_spins, n_samples)` | Sample random {-1, +1} configurations |

### `parisijax.core.solver`

| Function | Description |
|----------|-------------|
| `rs_free_energy(beta, h)` | Replica-symmetric free energy |
| `one_rsb_free_energy(q0, q1, m, beta, h)` | 1-step RSB free energy |
| `parisi_free_energy(q_raw, m_raw, beta, h)` | Full *k*-RSB Parisi free energy |
| `optimize_parisi(beta, h, k, ...)` | Optimize Parisi parameters, returns `ParisiResult` |
| `optimize_parisi_multistart(beta, ...)` | Multi-seed optimization |
| `find_critical_temperature()` | Locate the AT instability |

### `parisijax.core.mcmc`

| Function | Description |
|----------|-------------|
| `run_mcmc(key, J, beta, ...)` | Vectorized MCMC with N replicas |
| `parallel_tempering_step(key, spins, J, betas)` | Single PT step with replica exchange |

### `parisijax.analysis.overlap`

| Function | Description |
|----------|-------------|
| `compute_overlap(spins1, spins2)` | Pairwise overlap *q* |
| `compute_replica_overlap_matrix(spins)` | Full overlap matrix *Q*_αβ |
| `compute_edwards_anderson_parameter(overlaps)` | EA order parameter *q*_EA |
| `sample_overlap_distribution(key, J, beta, ...)` | Sample *P(q)* via MCMC |
| `theoretical_overlap_distribution(q, m)` | *P(q)* from Parisi solution |

### `parisijax.analysis.scaling`

| Function | Description |
|----------|-------------|
| `binder_cumulant(overlaps)` | Binder cumulant *g* |
| `susceptibility(overlaps, n_spins)` | Spin glass susceptibility *χ*_SG |
| `collect_observables(key, sizes, betas, ...)` | MCMC at all (*N*, *β*) points |
| `data_collapse(observables, sizes, betas, ...)` | Extract *β*_c and *ν* |
| `find_binder_crossing(observables, sizes, betas)` | Binder curve intersection |

---

## Examples

See the [`examples/`](examples/) directory:

1. **[01_quickstart.ipynb](examples/01_quickstart.ipynb)** — From Hamiltonian to phase transition (<2 min CPU)
2. **[02_parisi_solver.ipynb](examples/02_parisi_solver.ipynb)** — Differentiable Parisi solver deep dive (<5 min CPU)
3. **[03_finite_size_scaling.ipynb](examples/03_finite_size_scaling.ipynb)** — Monte Carlo to critical exponents (<15 min CPU)

---

## Citation

```bibtex
@software{parisijax2025,
  author  = {Seyoum, Nahom},
  title   = {{ParisiJAX}: GPU-Accelerated Spin Glass Physics},
  year    = {2025},
  url     = {https://github.com/nahomes-15/parisijax},
  version = {0.1.0}
}
```

---

## References

1. G. Parisi, "Infinite number of order parameters for spin-glasses," *Phys. Rev. Lett.* **43**, 1754 (1979).
2. G. Parisi, "A sequence of approximated solutions to the S-K model for spin glasses," *J. Phys. A* **13**, L115 (1980).
3. M. Mézard, G. Parisi, and M. A. Virasoro, *Spin Glass Theory and Beyond* (World Scientific, 1987).
4. M. Talagrand, "The Parisi formula," *Ann. Math.* **163**, 221 (2006).
5. F. Guerra, "Broken replica symmetry bounds in the mean field spin glass model," *Commun. Math. Phys.* **233**, 1 (2003).
6. J. R. L. de Almeida and D. J. Thouless, "Stability of the Sherrington-Kirkpatrick solution," *J. Phys. A* **11**, 983 (1978).

---

## License

MIT License. See [LICENSE](LICENSE).
