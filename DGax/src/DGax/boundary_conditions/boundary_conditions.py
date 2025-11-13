import equinox as eqx
import jax.numpy as jnp
from jax import Array
from typing import Protocol, Callable

class BoundaryCondition(eqx.Module, Protocol):
    """
    Protocol for boundary conditions.
    All BCs must implement __call__ that returns exterior state.
    """
    
    def __call__(
        self,
        u_interior: Array,      # [Nfp,] or scalar - interior state on face
        normal: Array,          # [2,] - face normal vector
        equations: eqx.Module,  # Equation object
        t: float                # Current time (for time-dependent BCs)
    ) -> Array:
        """Transform interior state to exterior state for boundary flux"""
        ...
        
class DirichletBC(eqx.Module):
    """
    Dirichlet boundary condition: u = u_boundary(x, y, t)
    
    For advection: u_exterior = 2*u_boundary - u_interior
    (upwind-biased for stability)
    """
    
    value_fn: Callable[[Array, Array, float], Array]  # (x, y, t) -> u
    
    def __call__(
        self,
        u_interior: Array,
        normal: Array,
        equations: eqx.Module,
        t: float
    ) -> Array:
        # Get boundary coordinates (need mesh info - see solver integration)
        # For now, assume value_fn handles this via closure
        u_boundary = self.value_fn(t)  # Simplified - see full example below
        
        # Upwind-extrapolated exterior state
        return 2 * u_boundary - u_interior


class OutflowBC(eqx.Module):
    """
    Outflow (zero-gradient) boundary: ∂u/∂n = 0
    
    Simply mirrors interior state: u_exterior = u_interior
    """
    
    def __call__(
        self,
        u_interior: Array,
        normal: Array,
        equations: eqx.Module,
        t: float
    ) -> Array:
        # Zero gradient: u_right = u_left
        return u_interior


class InflowBC(eqx.Module):
    """
    Inflow boundary: u = u_inflow(t)
    
    For advection with a·n < 0 (inflow), prescribe external value
    """
    
    value_fn: Callable[[float], Array]  # t -> u_inflow
    
    def __call__(
        self,
        u_interior: Array,
        normal: Array,
        equations: eqx.Module,
        t: float
    ) -> Array:
        # Check if this is truly inflow
        speed = jnp.dot(equations.velocity, normal)
        
        # If inflow (speed < 0), return prescribed value
        # If outflow (speed ≥ 0), use interior (upwind)
        prescribed = self.value_fn(t)
        return jnp.where(speed < 0, prescribed, u_interior)


class PeriodicBC(eqx.Module):
    """
    Periodic boundary: handled by mesh connectivity (mapP).
    This is a no-op - mapP already points to correct neighbor.
    """
    
    def __call__(self, u_interior: Array, *args) -> Array:
        raise NotImplementedError("Periodic BC handled by mesh connectivity")