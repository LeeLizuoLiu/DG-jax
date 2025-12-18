import equinox as eqx
from jax import Array
from typing import Protocol

class Equation(eqx.Module):
    """Protocol that all PDE equations must implement."""
    
    def flux(self, u: Array) -> Array:
        """
        Compute physical flux F(u).
        
        Args:
            u: Conservative variables, shape [..., n_vars]
        
        Returns:
            Flux tensor, shape [..., n_vars, dim]
        """
        ...
    
    def max_eigval(self, u: Array, normal: Array) -> Array:
        """
        Maximum absolute eigenvalue (wave speed) for Riemann solvers.
        
        Args:
            u: State variables, shape [..., n_vars]
            normal: Normal vector, shape [..., dim]
        
        Returns:
            Scalar wave speed per element, shape [...]
        """
        ...