"""ParisiJAX: JAX implementation of the Parisi RSB solution for spin glasses."""

__version__ = "0.1.0"

from parisijax.core import hamiltonian, solver, mcmc
from parisijax.analysis import overlap, scaling
from parisijax import viz

__all__ = ["hamiltonian", "solver", "mcmc", "overlap", "scaling", "viz"]
