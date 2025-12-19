# DGax/src/DGax/equations/Advection.py

import jax.numpy as jnp
from jax import Array
from typing import Tuple
from .equations import Equation

class Advection2D(Equation):
    """2D Linear Advection Equation: ∂u/∂t + a·∇u = 0
    
    A simple scalar linear equation ideal for testing framework.
    """
    
    velocity: Array  # Constant advection speed [vx, vy]
    
    def __init__(self, velocity: Tuple[float, float]):
        """Initialize with constant velocity vector (vx, vy)."""
        self.velocity = jnp.array(velocity, dtype=jnp.float64)
    
    def flux(self, u: Array) -> Array:
        """Flux: F(u) = a * u
        
        Args:
            u: Scalar field, shape [Np, K, 1] (last dim is n_vars)
        
        Returns:
            Flux tensor, shape [Np, K, 1, 2]
            Last dimension is [F, G] where F = vx*u, G = vy*u
        """
        # u has shape [Np, K, 1]
        # We need to return [Np, K, 1, 2]
        # where [:, :, 0, 0] = vx * u[:, :, 0]
        #       [:, :, 0, 1] = vy * u[:, :, 0]
        
        F = self.velocity[0] * u  # [Np, K, 1]
        G = self.velocity[1] * u  # [Np, K, 1]
        
        # Stack along new dimension
        return jnp.stack([F, G], axis=-1)  # [Np, K, 1, 2]
    
    def max_eigval(self, u: Array, normal: Array) -> Array:
        """Maximum wave speed = |a·n|
        
        Args:
            u: State (unused but kept for interface consistency)
            normal: Normal vector, shape [..., 2]
        
        Returns:
            Absolute wave speed, shape [...]
        """
        # Handle different input shapes for normal
        # normal can be [2,], [Nfp, 2], [Nfp, Nfaces, K, 2], etc.
        
        # Compute dot product along last dimension
        speed = jnp.sqrt(jnp.sum((self.velocity*normal)**2, axis=-1)) # jnp.sum(self.velocity * normal, axis=-1)
        return speed
    
    def max_wave_speed(self, u: Array, normal: Array) -> Array:
        """Alias for max_eigval for compatibility with LocalLaxFriedrichs
        
        Args:
            u: State (unused)
            normal: Normal vector, shape [..., 2]
        
        Returns:
            Absolute wave speed, shape [...]
        """
        return self.max_eigval(u, normal)