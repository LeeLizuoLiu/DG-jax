import jax.numpy as jnp
from jax import Array
from typing import Callable

def ssprk43_step(
    rhs_fn: Callable[[Array, float], Array],
    u: Array,
    t: float,
    dt: float
) -> Array:
    """
    Strong Stability Preserving RK4(3) - 5-stage, 4th order
    
    Optimal for hyperbolic PDEs with shocks (TVD/SSP property)
    """
    # Coefficients from Ketcheson 2008
    a21 = 1/2
    a32 = 1/2
    a43 = 1/2
    a54 = 1/2
    a65 = 1/6
    
    b1 = 1/6
    b2 = 0
    b3 = 0
    b4 = 0
    b5 = 0
    b6 = 5/6
    
    # Stage 1
    u1 = u + dt * rhs_fn(u, t)
    
    # Stage 2
    u2 = u + a21 * dt * (rhs_fn(u1, t + dt) - rhs_fn(u, t))
    
    # Stage 3
    u3 = u + a32 * dt * (rhs_fn(u2, t + dt/2) - rhs_fn(u, t))
    
    # Stage 4
    u4 = u + a43 * dt * (rhs_fn(u3, t + dt/2) - rhs_fn(u, t))
    
    # Stage 5
    u5 = u + a54 * dt * (rhs_fn(u4, t + dt/2) - rhs_fn(u, t))
    
    # Stage 6 (final)
    u6 = u + a65 * dt * (rhs_fn(u5, t + dt) - 
                         (b1*rhs_fn(u, t) + b6*rhs_fn(u5, t + dt)))
    
    return u6

def integrate_ssprk43(
    rhs_fn: Callable[[Array, float], Array],
    u0: Array,
    t_span: tuple[float, float],
    dt: float,
    n_steps: int
) -> tuple[Array, Array]:
    """
    Integrate ODE u'(t) = RHS(u, t) using SSPRK43
    
    Args:
        rhs_fn: Right-hand side function
        u0: Initial condition
        t_span: (t_start, t_end)
        dt: Time step size
        n_steps: Number of steps to take
        
    Returns:
        (final_solution, all_steps) - Solution at final time and all intermediate steps
    """
    t_start, t_end = t_span
    t = t_start
    
    # Storage for intermediate steps if needed
    solution_history = []
    u = u0
    
    for step in range(n_steps):
        u = ssprk43_step(rhs_fn, u, t, dt)
        t += dt
        solution_history.append(u)
        
        if t >= t_end:
            break
    
    return u, jnp.stack(solution_history)