from Euler2D import *
import pdb
def convergence_rate_test():
    """
    Test the convergence rate of the Euler 2D solver with respect to polynomial order.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from math import log
    
    # Define the mesh file
    mesh_file = "vortexA04.neu"
    
    # Define the polynomial orders to test
    orders = [1, 2, 3, 4, 5]
    
    # Set final time
    final_time = 0.1
    
    # Initialize arrays to store errors and DOFs
    errors = []
    dofs = []
    
    # Run simulations for each order
    for order in orders:
        print(f"\nRunning simulation with polynomial order {order}")
        
        # Initialize solver with current order
        EulerSolver = Euler_2D(mesh_file, order)
        
        # Calculate degrees of freedom
        dof = EulerSolver.K * EulerSolver.Np
        dofs.append(dof)
        
        # Compute initial condition
        Q_init = isentropic_vortex_ic_2d(EulerSolver.x, EulerSolver.y, 0)
        
        # Solve problem
        Q_final = EulerSolver.euler_2d(Q_init, final_time, isentropic_vortex_bc_2d)[-1]
        
        # Compute exact solution at final time
        Q_exact = isentropic_vortex_ic_2d(EulerSolver.x, EulerSolver.y, final_time)
        
        # Calculate L2 error for density (first component)
        error_density = calculate_l2_error(EulerSolver, Q_final[:,:,0], Q_exact[:,:,0])
        errors.append(error_density)
        
        print(f"Order: {order}, DOFs: {dof}, L2 Error: {error_density:.4e}")
    
    # Calculate convergence rates
    rates = []
    for i in range(1, len(orders)):
        # Calculate rate based on DOFs
        rate = -log(errors[i]/errors[i-1]) / log(dofs[i]/dofs[i-1])
        rates.append(rate)
        print(f"Order {orders[i-1]} → {orders[i]}: Convergence rate = {rate:.4f}")
    
def calculate_l2_error(solver, numerical, exact):
    """
    Calculate the L2 error between numerical and exact solutions.
    
    Args:
        solver: The Euler_2D solver instance
        numerical: Numerical solution
        exact: Exact solution
    
    Returns:
        L2 error
    """
    # Get mass matrix for integration
    M = np.linalg.inv(solver.V.T) @ np.linalg.inv(solver.V)
    
    # Calculate squared difference
    diff_squared = (numerical - exact) ** 2
    # Integrate the squared difference over the domain
    error_squared = 0
    for k in range(solver.K):
        local_error = diff_squared[:, k].T @ M @ diff_squared[:, k]
        error_squared += local_error
    
    # Take square root to get L2 error
    return np.sqrt(error_squared)

if __name__ == "__main__":
    convergence_rate_test()