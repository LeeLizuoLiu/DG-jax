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
    
    def conserved2primitive(self, Q: Array) -> Array:
        """
        Args:
            Q, conserved variable, shape [..., 4]
        
        Returns:
               primitive variable, shape [..., 4]
        """
        # Extract conserved variables
        rho  = Q[...,0]
        rhou = Q[...,1]
        rhov = Q[...,2]
        Ener = Q[...,3]
    
        # Compute primitive variables
        u = rhou / (jnp.abs(rho) + 1e-15)
        v = rhov / (jnp.abs(rho) + 1e-15)
        p = (self.gamma - 1) * (Ener - 0.5 * (rhou * u + rhov * v))
        return rho, u, v, p # jnp.stack([rho, u, v, p], axis=-1)

    def flux(self, Q: Array) -> Array:
        """Flux function for Euler equations
        
        Args:
            Q: shape [..., 4]
        
        Returns:
            Flux vector, shape [..., 4, 2]
        """
        rho, _, _, Ener = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]
        rho, u, v, p = self.conserved2primitive(Q) 
        # Compute flux functions using functional syntax
        F = jnp.stack([rho * u, rho * u * u + p, rho * v * u, u * (Ener + p)], axis=-1)
        G = jnp.stack([rho * v, rho * u * v, rho * v * v + p, v * (Ener + p)], axis=-1)
    
        return jnp.stack([F, G], axis=-1)
    
    def max_wave_speed(self, Q: Array, normal: Array) -> Array:
        """Maximum wave speed
        
        Args:
            u: Conserved variable [rho, rhou, rhov, Ener]
            normal: Normal vector [nx, ny]
        
        References: Trixi.jl, max_abs_speed
        Returns:
            Absolute wave speed
        """
        rho, u, v, p = self.conserved2primitive(Q)
        c = jnp.sqrt(jnp.abs(self.gamma * p / (jnp.abs(rho) + 1e-15)))
        proj_vel = normal[0] * u + normal[1] * v
        return jnp.abs(proj_vel) + c
