"""Animation and plotting utilities."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
import numpy as np


def plot_overlap_distribution(q_samples, beta, ax=None, bins=50):
    """Plot histogram of overlap distribution P(q).

    Args:
        q_samples: Array of overlap samples
        beta: Inverse temperature (for title)
        ax: Matplotlib axis (if None, creates new figure)
        bins: Number of histogram bins

    Returns:
        ax: The matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(np.array(q_samples), bins=bins, range=(-1, 1), density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Overlap q', fontsize=12)
    ax.set_ylabel('P(q)', fontsize=12)
    ax.set_title(f'Overlap Distribution at β = {beta:.2f}', fontsize=14)
    ax.grid(True, alpha=0.3)

    return ax


def plot_parisi_function(q_opt, m_opt, ax=None):
    """Plot the Parisi function q(x) as a staircase.

    The Parisi function is the inverse of x(q), which is piecewise constant.

    Args:
        q_opt: Optimal overlap parameters from Parisi solution
        m_opt: Optimal m parameters
        ax: Matplotlib axis (if None, creates new figure)

    Returns:
        ax: The matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Convert to numpy for plotting
    q_opt = np.array(q_opt)
    m_opt = np.array(m_opt)

    # Extend arrays to include boundaries
    q_extended = np.concatenate([[0.0], q_opt])
    m_extended = np.concatenate([[0.0], m_opt])

    # Plot as step function
    for i in range(len(q_opt)):
        ax.hlines(q_extended[i+1], m_extended[i], m_extended[i+1], colors='blue', linewidth=2)
        if i < len(q_opt) - 1:
            ax.vlines(m_extended[i+1], q_extended[i+1], q_extended[i+2], colors='blue', linewidth=2, linestyle='--')

    ax.set_xlabel('x (replica index parameter)', fontsize=12)
    ax.set_ylabel('q(x)', fontsize=12)
    ax.set_title('Parisi Function (Overlap Structure)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


def animate_rsb_transition(J, beta_range, n_frames=50, n_samples=500, n_steps=2000):
    """Create animation showing evolution of P(q) across temperatures.

    Args:
        J: Coupling matrix
        beta_range: (beta_min, beta_max) temperature range
        n_frames: Number of frames in animation
        n_samples: Number of overlap samples per frame
        n_steps: Number of MCMC steps per sample

    Returns:
        anim: Matplotlib animation object
    """
    import jax
    from parisijax.analysis.overlap import sample_overlap_distribution

    beta_min, beta_max = beta_range
    betas = np.linspace(beta_min, beta_max, n_frames)

    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame):
        """Update function for animation."""
        ax.clear()

        beta = betas[frame]

        # Sample overlaps at this temperature
        key = jax.random.PRNGKey(42 + frame)
        overlaps = sample_overlap_distribution(
            key, J, beta,
            n_samples=n_samples,
            n_steps=n_steps,
            burnin=500
        )

        # Plot histogram
        ax.hist(np.array(overlaps), bins=50, range=(-1, 1), density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Overlap q', fontsize=12)
        ax.set_ylabel('P(q)', fontsize=12)
        ax.set_title(f'Overlap Distribution | β = {beta:.2f} | T = {1/beta:.2f}', fontsize=14)
        ax.set_ylim(0, 5)
        ax.grid(True, alpha=0.3)

        return ax,

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)

    return anim


def plot_energy_trajectory(energy_trajectory, ax=None, n_show=10):
    """Plot energy evolution for multiple MCMC chains.

    Args:
        energy_trajectory: Energy values, shape (n_samples, n_steps)
        ax: Matplotlib axis (if None, creates new figure)
        n_show: Number of trajectories to show

    Returns:
        ax: The matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    energy_trajectory = np.array(energy_trajectory)
    n_samples, n_steps = energy_trajectory.shape

    # Show a subset of trajectories
    indices = np.linspace(0, n_samples - 1, n_show, dtype=int)

    for idx in indices:
        ax.plot(energy_trajectory[idx], alpha=0.5, linewidth=0.8)

    ax.set_xlabel('MCMC Step', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Energy Trajectories', fontsize=14)
    ax.grid(True, alpha=0.3)

    return ax


def plot_free_energy_comparison(betas, f_rs, f_parisi, ax=None):
    """Plot comparison of RS and Parisi free energies.

    Args:
        betas: Array of inverse temperatures
        f_rs: RS free energies
        f_parisi: Parisi free energies
        ax: Matplotlib axis (if None, creates new figure)

    Returns:
        ax: The matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    betas = np.array(betas)
    f_rs = np.array(f_rs)
    f_parisi = np.array(f_parisi)

    ax.plot(betas, f_rs, 'o-', label='Replica Symmetric (RS)', linewidth=2, markersize=6)
    ax.plot(betas, f_parisi, 's-', label='Parisi (Full RSB)', linewidth=2, markersize=6)
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='β_c ≈ 1.0')

    ax.set_xlabel('Inverse Temperature β', fontsize=12)
    ax.set_ylabel('Free Energy per Spin', fontsize=12)
    ax.set_title('Free Energy: RS vs Parisi Solution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return ax


def plot_overlap_matrix(overlap_matrix, ax=None):
    """Plot heatmap of replica overlap matrix.

    Args:
        overlap_matrix: Matrix of pairwise overlaps, shape (n_configs, n_configs)
        ax: Matplotlib axis (if None, creates new figure)

    Returns:
        ax: The matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    overlap_matrix = np.array(overlap_matrix)

    im = ax.imshow(overlap_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel('Configuration Index', fontsize=12)
    ax.set_ylabel('Configuration Index', fontsize=12)
    ax.set_title('Replica Overlap Matrix', fontsize=14)

    plt.colorbar(im, ax=ax, label='Overlap q')

    return ax
