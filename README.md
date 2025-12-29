# ParisiJax
**Differentiable Spin Glass Solvers in JAX**

[![JAX](https://img.shields.io/badge/JAX-Powered-blue.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Physics](https://img.shields.io/badge/Statistical-Physics-purple)](https://arxiv.org/abs/cond-mat/0000000)

**ParisiJax** is a high-performance, fully differentiable library for solving the **Sherrington-Kirkpatrick (SK)** model and other mean-field spin glasses. It connects the infinite-*N* theoretical limit (Parisi Formula) to finite-*N* reality (Monte Carlo simulations).

By leveraging JAX, this library allows researchers to:
1.  **Solve the Parisi PDE** numerically via auto-differentiable *k*-step Replica Symmetry Breaking (*k*-RSB).
2.  **Optimize thermodynamic quantities** (like Free Energy) via gradient descent on the functional order parameter *x(q)*.
3.  **Simulate massive systems** (*N* > 10,000) on GPUs using `vmap`-accelerated MCMC.

---

## 1. The Mathematical Theory

### The Hamiltonian
The SK model describes *N* Ising spins $\sigma_i \in \{-1, 1\}$ with quenched Gaussian disorder:

$$
H_N(\boldsymbol{\sigma}) = -\frac{1}{\sqrt{N}} \sum_{1 \leq i < j \leq N} J_{ij} \sigma_i \sigma_j - h \sum_{i=1}^N \sigma_i
$$

where $J_{ij} \sim \mathcal{N}(0, 1)$ are i.i.d. random couplings.

### The Parisi Variational Formula
In the thermodynamic limit (*N* → ∞), the quenched free energy density converges to the **Parisi Formula**:

$$
f(\beta, h) = \inf_{x \in \mathcal{X}} \mathcal{P}[x]
$$

where the functional $\mathcal{P}[x]$ is defined by the solution to a nonlinear partial differential equation (PDE). Let *x* : [0, 1] → [0, 1] be a non-decreasing function (the order parameter). We define the function $\Phi(q, y)$ on $[0, 1] \times \mathbb{R}$ as the solution to the **Parisi Backward PDE**:

$$
\frac{\partial \Phi}{\partial q} + \frac{1}{2} \frac{\partial^2 \Phi}{\partial y^2} + \frac{1}{2} x(q) \left( \frac{\partial \Phi}{\partial y} \right)^2 = 0
$$

with the terminal condition at *q* = 1:

$$
\Phi(1, y) = \log \cosh (\beta y)
$$

The free energy is then given by:

$$
\mathcal{P}[x] = -\frac{\beta}{4} \left( 1 - \int_0^1 x(q) \, dq \right) - \frac{1}{\beta} \Phi(0, h)
$$

### The k-RSB Approximation
Numerically, we cannot solve for a continuous *x(q)* directly. Instead, we approximate *x(q)* as a step function with *k* steps. This discretizes the PDE into a recursive integral equation.

Let $0 = m_0 < m_1 < \dots < m_k = 1$ be the values of *x(q)* in each interval $[q_i, q_{i+1}]$. The PDE becomes a sequence of Gaussian convolutions:

$$
f_i(y) = \frac{1}{m_i} \log \int_{\mathbb{R}} \mathcal{D}z \, \exp\left( m_i f_{i+1}\left( y + z\sqrt{q_{i+1} - q_i} \right) \right)
$$

ParisiJax implements this recursion using `jax.lax.scan` and differentiable Gaussian quadrature, allowing gradients to flow back to the parameters $\{m_i, q_i\}$.

---

## 2. Installation

```
```bash
pip install parisijax
