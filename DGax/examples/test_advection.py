# file: test_simple_periodic.py
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
# Add this import at the top of your file
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter

from DGax.mesh.TriMesh import TriMesh
from DGax.equations.Advection import Advection2D
from DGax.riemann_solvers.LocalLaxFriedrichs import local_lax_friedrichs_matlab
from DGax.solver import DGSolver
from DGax.integrators.SSPRK import rk4_step
import jax 
import pdb
jax.config.update("jax_enable_x64", True)   
# jax.config.update("jax_platform_name", "cpu")

def initial_condition_simple(x, y, t=0):
    """Simple sine wave initial condition."""
    return jnp.sin(6 * jnp.pi * (x - 2*t)) * jnp.sin(4 * jnp.pi * (y - t)) + 1

def run_simple_test():
    """Run a simple periodic advection test."""
    print("Running simple periodic advection test...")
    
    # Parameters
    order = 2 
    h = 0.01
    velocity = (2.0, 1)
    dt = 0.3/3000
    n_steps = 3000 
    
    # Create periodic mesh
    print("Generating periodic mesh...")
    mesh = TriMesh.from_gmsh_periodic(
        order=order,
        h=h,
        domain_size=1.0
    )
    
    print(f"Mesh created: K={mesh.K}, Np={mesh.Np}")
    
    # Create equation
    eqn = Advection2D(velocity=velocity)
    
    # Initial condition
    u0 = initial_condition_simple(mesh.x, mesh.y)
    u0 = u0[..., jnp.newaxis]  # Add variable dimension
    
    # Create solver (no boundary conditions for periodic)
    solver = DGSolver(
        equations=eqn,
        riemann_solver=local_lax_friedrichs_matlab,
        mesh=mesh,
        boundary_conditions={}
    )
    
    # Time stepping
    u = u0.copy()
    t = 0.0
    

    # Store solution snapshots during time integration
    snapshots = []
    exact_snapshots = []
    snapshot_times = []
    
    u = u0.copy()
    t = 0.0
    
    # Store initial condition
    snapshots.append(u.copy())
    exact_snapshots.append(u.copy())
    snapshot_times.append(t)
    
    print(f"Running time integration: dt={dt:.6f}, n_steps={n_steps}")
    
    # Number of snapshots to save for animation
    num_snapshots = min(100, n_steps)  # Max 100 frames
    save_interval = max(1, n_steps // num_snapshots)
    
    for step in range(n_steps):
        u = rk4_step(solver.rhs, u, t, dt)
        t += dt
        
        if (step + 1) % save_interval == 0:
            snapshots.append(u.copy())
            snapshot_times.append(t)
            exact_snapshots.append(initial_condition_simple(mesh.x, mesh.y, t))
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{n_steps}, t={t:.4f}")
    
    # Store final solution if not already stored
    # if len(snapshots) == 0 or snapshot_times[-1] < final_time:
    #     snapshots.append(u.copy())
    #     exact_snapshots.append(initial_condition_simple(mesh.x, mesh.y, t))
    #     snapshot_times.append(t)
    
    print(f"Saved {len(snapshots)} snapshots for animation")
    
    # Create animation
    # print("\nCreating animation...")
    
    # # Prepare data for animation
    # x_flat = mesh.x.flatten('F')
    # y_flat = mesh.y.flatten('F')

    # # Find min and max for consistent colorbar
    # all_data = np.array([snap[..., 0].flatten('F') for snap in snapshots]) - np.array([snap.flatten('F') for snap in exact_snapshots])
    # vmin, vmax = all_data.min(), all_data.max()
    
    # # Create figure for animation
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # # Initialize scatter plots
    # scatter1 = ax1.scatter(x_flat, y_flat, 
    #                       c=exact_snapshots[0].flatten('F') - snapshots[0][..., 0].flatten('F'), 
    #                       cmap='viridis', s=20, vmin=vmin, vmax=vmax)
    # scatter2 = ax2.scatter(x_flat, y_flat,
    #                       c=exact_snapshots[0].flatten('F') - snapshots[0][..., 0].flatten('F'),
    #                       cmap='viridis', s=20, vmin=vmin, vmax=vmax)
    
    # # Set up subplots
    # for ax in [ax1, ax2]:
    #     ax.set_aspect('equal')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 1)
    
    # ax1.set_title('Error Evolution (Animation)')
    # ax2.set_title(f'Velocity: ({velocity[0]}, {velocity[1]})')
    
    # # Add colorbar
    # cbar = fig.colorbar(scatter1, ax=[ax1, ax2], orientation='horizontal', 
    #                    fraction=0.05, pad=0.1)
    # cbar.set_label('u(x,y,t)')
    
    # # Add time text
    # time_text = ax1.text(0.02, 0.95, f'Time: {snapshot_times[0]:.3f}', 
    #                     transform=ax1.transAxes, fontsize=12,
    #                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # # Add velocity arrow
    # vx, vy = velocity
    # arrow = ax2.quiver(0.5, 0.5, vx, vy, color='red', scale=10,
    #                   label=f'Velocity: ({vx}, {vy})')
    # ax2.legend(loc='upper left')
    
    # # Animation update function
    # def update(frame):
    #     """Update function for animation"""
    #     # Update scatter plot data
    #     scatter1.set_array(exact_snapshots[frame].flatten('F') - snapshots[frame][..., 0].flatten('F'))
    #     scatter2.set_array(exact_snapshots[frame].flatten('F') - snapshots[frame][..., 0].flatten('F'))
    #     
    #     # Update time text
    #     time_text.set_text(f'Time: {snapshot_times[frame]:.3f}')
    #     
    #     return scatter1, scatter2, time_text
    
    # Create animation
    # anim = FuncAnimation(fig, update, frames=len(snapshots),
    #                     interval=100, blit=True)
    
    # # Save animation as video
    # print("Saving animation as MP4...")
    # writer = FFMpegWriter(fps=10, metadata=dict(artist='DG Solver'), bitrate=1800)
    # anim.save('advection_animation.mp4', writer=writer)
    
    # Save final frame as static image
    plt.figure(figsize=(12, 5))
    
    # Initial condition
    # plt.subplot(1, 2, 1)
    # plt.scatter(x_flat, y_flat, c=snapshots[0][..., 0].flatten('F'), 
    #            cmap='viridis', s=20)
    # plt.gca().set_aspect('equal')
    # plt.title('Initial Condition (t=0)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar(label='u(x,y,0)')
    
    # # Final solution
    # plt.subplot(1, 2, 2)
    # plt.scatter(x_flat, y_flat, c=snapshots[-1][..., 0].flatten('F'), 
    #            cmap='viridis', s=20)
    # plt.gca().set_aspect('equal')
    # plt.title(f'Final Solution at t={final_time:.3f}')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar(label=f'u(x,y,{final_time:.3f})')
    
    # plt.tight_layout()
    # plt.savefig('simple_periodic_test.png', dpi=150, bbox_inches='tight')
    # plt.show()
    
    # Check mass conservation (should be conserved with periodic BCs)
    exact_mass = jnp.mean(exact_snapshots[-1])
    final_mass = jnp.mean(u[..., 0])
    mass_change = abs(final_mass - exact_mass) / abs(exact_mass)
    
    print(f"\nMass conservation check:")
    print(f"  Exact mass: {exact_mass:.6e}")
    print(f"  Final mass:   {final_mass:.6e}")
    print(f"  Relative change: {mass_change:.6e}")
    
    if mass_change < 1e-10:
        print("  ✓ Mass conserved (good for periodic BCs)")
    else:
        print(f"  ⚠ Mass not perfectly conserved (change: {mass_change:.2e})")
    
    # Also plot mass evolution over time
    plt.figure(figsize=(10, 6))
    
    # Calculate mass at each snapshot
    masses = jnp.array([jnp.mean(snap[..., 0]) for snap in snapshots])
    exact_masses = jnp.array([jnp.mean(snap) for snap in exact_snapshots])
    times = snapshot_times
    
    plt.plot(times, masses - exact_masses, 'b-', linewidth=2, label='Total Mass')
    # plt.axhline(y=exact_mass, color='r', linestyle='--', label='Exact Mass')
    
    plt.xlabel('Time')
    plt.ylabel('Mass')
    plt.title('Mass Conservation Check')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text box with mass change info
    textstr = f'Exact Mass: {exact_mass:.6e}\nFinal Mass: {final_mass:.6e}\nRelative Change: {mass_change:.2e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                  fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('mass_conservation.png', dpi=150, bbox_inches='tight')
    plt.show()

    # After time integration, compute mass conservation properly
    print(f"\n{'='*60}")
    print("MASS CONSERVATION (PROPER DG INTEGRATION)")
    print(f"{'='*60}")

    # Compute using DGSolver's method
    mass_info = solver.compute_mass_conservation_error(u, u0)

    print(f"Initial mass (DG integral): {mass_info['initial_mass']:.12e}")
    print(f"Final mass (DG integral):   {mass_info['current_mass']:.12e}")
    print(f"Absolute error: {mass_info['absolute_error']:.12e}")
    print(f"Relative error: {mass_info['relative_error']:.12e}")

    # Compare with simple mean method (for reference)
    simple_initial = jnp.mean(u0[..., 0])
    simple_final = jnp.mean(u[..., 0])
    simple_error = abs(simple_final - simple_initial) / abs(simple_initial)

    print(f"\n{'='*60}")
    print("COMPARISON WITH SIMPLE MEAN METHOD")
    print(f"{'='*60}")
    print(f"Simple mean initial: {simple_initial:.12e}")
    print(f"Simple mean final:   {simple_final:.12e}")
    print(f"Simple relative error: {simple_error:.12e}")

    print(f"\nDifference between methods:")
    print(f"  DG integral vs simple mean (initial): {abs(mass_info['initial_mass'] - simple_initial):.12e}")
    print(f"  DG integral vs simple mean (final):   {abs(mass_info['current_mass'] - simple_final):.12e}")

    # Plot mass evolution during time integration
    print(f"\n{'='*60}")
    print("MASS EVOLUTION DURING TIME INTEGRATION")
    print(f"{'='*60}")

    if 'snapshots' in locals() and len(snapshots) > 0:
        masses_dg = []
        masses_simple = []

        for i, snap in enumerate(snapshots):
            # DG integral method
            mass_dg = solver.compute_conserved_quantity(snap, 0)
            masses_dg.append(mass_dg)

            # Simple mean method
            mass_simple = jnp.mean(snap[..., 0])
            masses_simple.append(mass_simple)

        plt.figure(figsize=(12, 5))

        # Plot both methods
        plt.subplot(1, 2, 1)
        plt.plot(snapshot_times, masses_dg, 'b-', linewidth=2, label='DG Integral')
        plt.plot(snapshot_times, masses_simple, 'r--', linewidth=2, label='Simple Mean')
        plt.axhline(y=masses_dg[0], color='k', linestyle=':', alpha=0.5, label='Initial')
        plt.xlabel('Time')
        plt.ylabel('Mass')
        plt.title('Mass Evolution Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot error relative to initial
        plt.subplot(1, 2, 2)
        error_dg = [(m - masses_dg[0])/masses_dg[0] for m in masses_dg]
        error_simple = [(m - masses_simple[0])/masses_simple[0] for m in masses_simple]

        plt.semilogy(snapshot_times, np.abs(error_dg), 'b-', linewidth=2, label='DG Integral Error')
        plt.semilogy(snapshot_times, np.abs(error_simple), 'r--', linewidth=2, label='Simple Mean Error')
        plt.xlabel('Time')
        plt.ylabel('Relative Error')
        plt.title('Mass Conservation Error')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig('mass_evolution_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return mesh, u0, u, snapshots, snapshot_times

if __name__ == "__main__":
    run_simple_test()