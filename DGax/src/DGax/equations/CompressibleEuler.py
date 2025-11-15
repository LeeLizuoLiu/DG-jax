import jax.numpy as jnp
from .equations import Equation
from jax import Array
from typing import Tuple, Float  


class Euler2D(Equation):
    """2D Compressible Euler Equation: ∂u/∂t +∇·F(u) = 0
    """
    
    gamma: Float  
    
    def __init__(self, gamma: Float):
        """Initialize with constant velocity vector (vx, vy)."""
        self.gamma = jnp.array(gamma, dtype=jnp.float64)
    
    def flux(self, u: Array) -> Array:
        """Flux: F(u) = a * u
        
        Args:
            u: Scalar field, shape [..., 1] or [...]
        
        Returns:
            Flux vector, shape [..., 2]
        """
        # Ensure u has a variable dimension for broadcasting
        u_ = jnp.atleast_1d(u)
        return u_[..., None] * self.velocity
    
    def max_eigval(self, u: Array, normal: Array) -> Array:
        """Maximum wave speed = |a·n|
        
        Args:
            u: State (unused but kept for interface consistency)
            normal: Normal vector [nx, ny]
        
        Returns:
            Absolute wave speed, scalar or broadcasted shape
        """
        speed = jnp.dot(self.velocity, normal)
        return jnp.abs(speed)
