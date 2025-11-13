import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from DGax.equations.Advection import Advection2D
from DGax.riemann_solvers.upwind import upwind_flux
from DGax.solver import DGSolver
from DGax.integrators.SSPRK import integrate_ssprk43
from DGax.mesh import TriMesh
import warnings
warnings.filterwarnings("ignore")

@dataclass
class SimulationConfig:
    """Problem configuration"""
    mesh_file: str = "vortexA04.neu"
    order: int = 3
    velocity: tuple = (1.0, 0.5)
    t_final: float = 1.0
    cfl: float = 0.5

def initialize_solution(mesh, config: SimulationConfig) -> jax.Array:
    """
    Gaussian pulse initial condition
    """
    # Gaussian centered at (0.3, 0.3)
    x0, y0 = 0.3, 0.3
    sigma = 0.05
    u0 = jnp.exp(-((mesh.x - x0)**2 + (mesh.y - y0)**2) / (2 * sigma**2))
    
    # Flatten to [Np, K] format
    return u0.reshape(mesh.x.shape)

def main():
    """Main simulation loop"""
    
    # --- Configuration ---
    config = SimulationConfig()
    
    # --- Step 1: Build mesh (one-time cost) ---
    print("Building mesh...")
    mesh = TriMesh.from_gambit(config.mesh_file, order=config.order)
    print(f"Mesh: {mesh.n_elements} elements, {mesh.Np} nodes/element")
    
    # --- Step 2: Initialize solver ---
    equations = Advection2D(config.velocity)
    solver = DGSolver(
        equations=equations,
        riemann_solver=upwind_flux,
        mesh=mesh,
        cfl=config.cfl
    )
    
    # --- Step 3: Initial condition ---
    u0 = initialize_solution(mesh, config)
    
    # --- Step 4: Compute stable time step ---
    dt = solver.compute_dt(u0)
    n_steps = int(config.t_final / dt)
    print(f"CFL dt: {dt:.5f}, steps: {n_steps}")
    
    # --- Step 5: Time integration ---
    print("Starting time integration...")
    final_u, history = integrate_ssprk43(
        rhs_fn=solver.rhs,
        u0=u0,
        t_span=(0.0, config.t_final),
        dt=dt,
        n_steps=n_steps
    )
    
    # --- Step 6: Post-processing ---
    print("Computing diagnostics...")
    
    # Mass conservation check
    mass_0 = jnp.sum(u0 * mesh.J)
    mass_final = jnp.sum(final_u * mesh.J)
    mass_error = (mass_final - mass_0) / mass_0
    
    # L2 error (if analytic solution available)
    def analytic_solution(x, y, t):
        """Gaussian moving with velocity"""
        x0, y0 = 0.3 + config.velocity[0] * t, 0.3 + config.velocity[1] * t
        sigma = 0.05
        return jnp.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    u_exact = analytic_solution(mesh.x, mesh.y, config.t_final)
    l2_error = jnp.sqrt(jnp.sum((final_u - u_exact)**2 * mesh.J))
    
    print(f"Mass conservation error: {mass_error:.2e}")
    print(f"L2 error: {l2_error:.2e}")
    
    # --- Step 7: Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Initial condition
    im0 = axes[0].contourf(mesh.x, mesh.y, u0, levels=20, cmap='viridis')
    axes[0].set_title("Initial Condition (t=0)")
    fig.colorbar(im0, ax=axes[0])
    
    # Final numerical solution
    im1 = axes[1].contourf(mesh.x, mesh.y, final_u, levels=20, cmap='viridis')
    axes[1].set_title(f"Numerical Solution (t={config.t_final})")
    fig.colorbar(im1, ax=axes[1])
    
    # Exact solution
    im2 = axes[2].contourf(mesh.x, mesh.y, u_exact, levels=20, cmap='viridis')
    axes[2].set_title("Exact Solution")
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("advection_results.png", dpi=150)
    print("Saved visualization to advection_results.png")
    
    return solver, history

if __name__ == "__main__":
    # Enable JAX double precision
    jax.config.update("jax_enable_x64", True)
    
    # Run simulation
    solver, history = main()
    
    # Optional: Profile the RHS evaluation
    # print("Profiling RHS...")
    # %timeit solver.rhs(solver.mesh.x, 0.0)