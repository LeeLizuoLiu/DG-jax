import jax.numpy as jnp
from jax import Array
from ..equations.Advection import Advection2D

def upwind_solver(
    u_left: Array,
    u_right: Array,
    normal: Array,
    equations: Advection2D
) -> Array:
    """Exact upwind Riemann solver for advection equation.
    
    Characteristic speed a·n:
    - If > 0: information from left, take u_left
    - If < 0: information from right, take u_right
    
    Args:
        u_left: Left state, shape [...]
        u_right: Right state, shape [...]
        normal: Face normal vector [nx, ny]
        equations: Equation object containing velocity a
    
    Returns:
        Numerical flux, shape [...]
    """
    speed = jnp.dot(equations.velocity, normal)  # a·n
    
    # Upwind selection
    upwind_state = jnp.where(speed >= 0, u_left, u_right)
    
    # Return flux = (a·n) * u_upwind
    return speed * upwind_state