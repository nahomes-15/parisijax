# ParisiJax

**Differentiable Spin Glass Solvers in JAX**

**ParisiJax** is a high-performance, fully differentiable library for solving the **Sherrington-Kirkpatrick (SK)** model. It bridges the gap between the infinite-$N$ theoretical limit (Parisi Formula) and finite-$N$ reality (Monte Carlo simulations).

By leveraging JAX, this library allows researchers to:

1. **Solve the Parisi PDE** numerically via auto-differentiable $k$-step Replica Symmetry Breaking ($k$-RSB).
2. **Optimize thermodynamic quantities** (like Free Energy) via gradient descent on the functional order parameter $q(x)$.
3. **Simulate massive systems** ($N \sim 10^3$) on GPUs using `vmap`-accelerated MCMC.

---

## 1. The Mathematical Theory

### The Hamiltonian

The SK model describes $N$ Ising spins $\sigma_i \in \{\pm 1\}$ with quenched Gaussian disorder:

$$
H = -\frac{1}{\sqrt{N}} \sum_{i<j} J_{ij} \sigma_i \sigma_j - h \sum_i \sigma_i
$$

where $J_{ij} \sim \mathcal{N}(0, 1)$ are i.i.d. random couplings.

### The Parisi Variational Formula

In the thermodynamic limit ($N \to \infty$), the quenched free energy density converges to the **Parisi Formula**:

$$
f = \inf_{q(x)} \Phi[q(x)]
$$

where the functional $\Phi$ is defined by the solution to a nonlinear partial differential equation (PDE). Let $q(x): [0,1] \to [0, q_{\max}]$ be a non-decreasing function (the order parameter).

We define the function $\Phi(x, y)$ on $x \in [0,1], y \in \mathbb{R}$ as the solution to the **Parisi Backward PDE**:

$$
\frac{\partial \Phi}{\partial x} = \frac{1}{2} q'(x) \frac{\partial^2 \Phi}{\partial y^2}
$$

with the terminal condition at $x=1$:

$$
\Phi(1, y) = \log 2 \cosh(\beta y)
$$

The free energy is then given by:

$$
f = -\frac{\beta q_{\max}}{4} + \frac{1}{\beta} \Phi(0, h)
$$

### The $k$-RSB Approximation

Numerically, we cannot solve for a continuous $q(x)$ directly. Instead, we approximate $q(x)$ as a step function with $k$ steps. This discretizes the PDE into a recursive integral equation.

Let $q_0 = 0 < q_1 < \cdots < q_k$ be the values of $q$ in each interval $[x_r, x_{r+1}]$. The PDE becomes a sequence of Gaussian convolutions:

$$
\Phi_r(y) = \frac{1}{m_r} \log \int \mathcal{D}z \, \exp\left[ m_r \Phi_{r+1}(y + \sqrt{\Delta q_r} z) \right]
$$

ParisiJax implements this recursion using `jax.lax.scan` and differentiable Gaussian quadrature, allowing gradients to flow back to the parameters $\{q_r, m_r\}$.

---

## 2. Installation

```bash
pip install parisijax
```

**Requirements:** `jax`, `jaxlib`, `optax`, `numpy`, `scipy`.
*Note: For GPU acceleration, ensure you have the correct CUDA-enabled JAX version installed.*

---

## 3. Usage: The Differentiable Solver

We can find the stable solution by minimizing the free energy with respect to the order parameters. Unlike traditional analytical derivations (which are painful for $k > 1$), we use **gradient descent**.

```python
import jax
import jax.numpy as jnp
from parisijax.core import solver

# 1. Initialize parameters for 10-step RSB
beta = 2.0  # Low temperature (Spin Glass Phase)
k_steps = 10
key = jax.random.PRNGKey(0)

# 2. Optimize the Parisi Functional
# This automatically discovers the shape of x(q)
params, free_energy, history = solver.optimize_parisi_structure(
    key,
    beta=beta,
    h=0.0,
    k=k_steps,
    learning_rate=1e-3,
    n_iters=5000
)

# 3. Extract the Order Parameter x(q)
q_grid = params['q']  # The overlap values
x_grid = params['m']  # The cumulative probability
# Plotting (q_grid, x_grid) reveals the phase transition structure.
```

### Why is this "Differentiable"?

Because the solver is pure JAX, you can differentiate the free energy with respect to physical parameters. For example, to compute the **Internal Energy** $U = \partial(\beta f)/\partial \beta$:

```python
def compute_internal_energy(beta, h, params):
    # Differentiate the solver solution w.r.t beta
    f_fn = lambda b: solver.parisi_free_energy(params['q'], params['m'], b, h)
    return jax.grad(f_fn)(beta)
```

---

## 4. Usage: GPU-Accelerated MCMC

To verify theoretical predictions, we simulate finite-$N$ instances. `ParisiJax` uses `jax.vmap` to simulate thousands of replicas in parallel on a single GPU.

```python
from parisijax.core import mcmc, hamiltonian

N = 1000
n_replicas = 4096  # Massive parallelization
beta = 1.5

# 1. Generate Couplings (J_ij)
key, subkey = jax.random.split(key)
J = hamiltonian.generate_sk_couplings(subkey, N)

# 2. Run Parallel Tempering or Metropolis
# Returns spins for all 4096 replicas
final_spins = mcmc.run_parallel_chains(
    key,
    J,
    beta=beta,
    n_steps=10_000,
    batch_size=n_replicas
)

# 3. Compute Overlap Distribution P(q)
# We compute q between pairs of replicas
q_overlaps = mcmc.compute_overlaps(final_spins)
```

---

## 5. Implementation Details

### Numerical Stability via Log-Sum-Exp

The core recursion involves terms like $\log \sum_i \exp(m_r \Phi_i)$. Naive implementation leads to overflows at low temperatures ($\beta > 2$). We utilize the **Log-Sum-Exp** trick stabilized by the maximum value in the Gaussian integral kernel:

$$
\log \sum_i w_i e^{m_r \Phi_i} = m_{\max} + \log \sum_i w_i e^{m_r (\Phi_i - \Phi_{\max})}
$$

### The De Almeida-Thouless (AT) Condition

The library includes a utility to check the stability of the Replica Symmetric (RS) solution. The RS solution is stable only if:

$$
\beta^2 (1 - q^2) \leq 1
$$

If this condition is violated (The AT line), the solver automatically switches to RSB mode.

---

## References

1. **Parisi, G.** (1980). *A sequence of approximate solutions to the S-K model for spin glasses*. J. Phys. A.
2. **Talagrand, M.** (2006). *The Parisi Formula*. Annals of Mathematics. (The rigorous proof).
3. **Mezard, M., Parisi, G., & Virasoro, M.** (1987). *Spin Glass Theory and Beyond*. World Scientific.

## License

MIT
