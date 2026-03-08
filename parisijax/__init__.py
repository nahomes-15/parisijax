"""ParisiJAX: JAX implementation of the Parisi RSB solution for spin glasses."""

__version__ = "0.1.0"

from parisijax import viz
from parisijax.analysis import overlap, research, scaling
from parisijax.core import hamiltonian, mcmc, solver

__all__ = ["hamiltonian", "solver", "mcmc", "overlap", "research", "scaling", "viz"]
