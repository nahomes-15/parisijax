# ParisiJax
**Differentiable Spin Glass Solvers in JAX**

[![JAX](https://img.shields.io/badge/JAX-Powered-blue.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://python.org)

**ParisiJax** is a high-performance, fully differentiable library for solving the **Sherrington-Kirkpatrick (SK)** model and other mean-field spin glasses. It bridges the gap between the infinite-*N* theoretical limit (Parisi Formula) and finite-*N* reality (Monte Carlo simulations).

By leveraging JAX, this library allows researchers to:
1. **Solve the Parisi PDE** numerically via auto-differentiable *k*-step Replica Symmetry Breaking (*k*-RSB).
2. 2. **Optimize thermodynamic quantities** (like Free Energy) via gradient descent on the functional order parameter *x(q)*.
   3. 3. **Simulate massive systems** (*N* > 10,000) on GPUs using `vmap`-accelerated MCMC.
     
      4. ---
     
      5. ## Installation
     
      6. ```bash
         pip install parisijax
         ```

         Or install from source:

         ```bash
         git clone https://github.com/nahomes-15/parisijax.git
         cd parisijax
         pip install -e .
         ```

         ### Dependencies
         - JAX >= 0.4.0
         - - NumPy >= 1.20
           - - SciPy >= 1.7
            
             - ---

             ## Quick Start

             ### Computing Free Energy with Replica Symmetry (RS)

             ```python
             import jax.numpy as jnp
             from parisijax.core.solver import SKSolver

             # Create solver at inverse temperature beta=1.0
             solver = SKSolver(beta=1.0, h=0.0)

             # Compute RS free energy
             f_rs = solver.rs_free_energy()
             print(f"RS Free Energy: {f_rs:.6f}")

             # Check AT stability (is RS solution stable?)
             is_stable = solver.at_stability()
             print(f"AT Stable: {is_stable}")
             ```

             ### 1-RSB Free Energy

             ```python
             # For beta > 1, RS becomes unstable. Use 1RSB:
             solver = SKSolver(beta=1.5, h=0.0)

             # Compute 1RSB free energy with breakpoint parameters
             # q1: first overlap, m: Parisi parameter
             f_1rsb = solver.one_rsb_free_energy(q1=0.5, m=0.7)
             print(f"1RSB Free Energy: {f_1rsb:.6f}")
             ```

             ### MCMC Simulation

             ```python
             import jax
             from parisijax.core.mcmc import SKModel

             # Create SK model with N=1000 spins
             key = jax.random.PRNGKey(42)
             model = SKModel(N=1000, beta=1.0, h=0.0, key=key)

             # Run parallel tempering MCMC
             samples, energies = model.sample(
                 n_samples=10000,
                 n_chains=4,
                 warmup=1000
             )

             # Compute overlap distribution
             from parisijax.core.mcmc import overlap_distribution
             q_values, p_q = overlap_distribution(samples)
             ```

             ---

             ## The Mathematical Theory

             ### The Hamiltonian
             The SK model describes *N* Ising spins with quenched Gaussian disorder:

             $$
             H_N(\boldsymbol{\sigma}) = -\frac{1}{\sqrt{N}} \sum_{1 \leq i < j \leq N} J_{ij} \sigma_i \sigma_j - h \sum_{i=1}^N \sigma_i
             $$

             where the couplings are i.i.d. standard normal random variables.

             ### The Parisi Variational Formula
             In the thermodynamic limit (*N* → ∞), the quenched free energy density converges to the **Parisi Formula**:

             $$
             f(\beta, h) = \inf_{x \in \mathcal{X}} \mathcal{P}[x]
             $$

             where the functional is defined by the solution to the **Parisi Backward PDE**:

             $$
             \frac{\partial \Phi}{\partial q} + \frac{1}{2} \frac{\partial^2 \Phi}{\partial y^2} + \frac{1}{2} x(q) \left( \frac{\partial \Phi}{\partial y} \right)^2 = 0
             $$

             with terminal condition at *q* = 1:

             $$
             \Phi(1, y) = \log \cosh (\beta y)
             $$

             ### The k-RSB Approximation
             Numerically, we approximate *x(q)* as a step function with *k* steps. ParisiJax implements this recursion using `jax.lax.scan` and differentiable Gaussian quadrature.

             ---

             ## API Reference

             ### `SKSolver`

             | Method | Description |
             |--------|-------------|
             | `rs_free_energy()` | Replica Symmetric free energy |
             | `at_stability()` | de Almeida-Thouless stability check |
             | `one_rsb_free_energy(q1, m)` | 1-RSB free energy |
             | `ground_state_energy()` | T→0 ground state energy |
             | `high_temp_free_energy()` | High temperature (β→0) limit |

             ### `SKModel`

             | Method | Description |
             |--------|-------------|
             | `sample(n_samples, n_chains, warmup)` | Run MCMC sampling |
             | `energy(spins)` | Compute energy of configuration |

             ### Utility Functions

             | Function | Description |
             |----------|-------------|
             | `compute_overlap(s1, s2)` | Overlap between two configurations |
             | `overlap_distribution(samples)` | P(q) from MCMC samples |

             ---

             ## Known Theoretical Results

             The implementation validates against known analytical results:

             | Quantity | Value | Reference |
             |----------|-------|-----------|
             | Ground state energy | -0.7633... | Parisi (1980) |
             | Critical temperature | β_c = 1 | SK (1975) |
             | RS free energy (β=1, h=0) | -0.75 | Exact |

             ---

             ## Citation

             If you use ParisiJax in your research, please cite:

             ```bibtex
             @software{parisijax2024,
               title={ParisiJax: Differentiable Spin Glass Solvers in JAX},
               author={nahomes-15},
               year={2024},
               url={https://github.com/nahomes-15/parisijax}
             }
             ```

             ---

             ## License

             MIT License - see [LICENSE](LICENSE) for details.

             ## Contributing

             Contributions are welcome! Please feel free to submit a Pull Request.
