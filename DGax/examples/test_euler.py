import numpy as np
from DGax.mesh.TriMesh import TriMesh
from DGax.integrators.SSPRK import rk4_step
from DGax.equations.CompressibleEuler import Euler2D
from DGax.solver import DGSolver
from DGax.riemann_solvers.LocalLaxFriedrichs import local_lax_friedrichs_matlab
import jax.numpy as jnp
import jax
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

jax.config.update("jax_enable_x64", True)  # Enable double precision
jax.config.update("jax_debug_nans", True)  # Enable NaN debugging
from functools import partial

import pdb

def isentropic_vortex_ic_2d(Q, n, x, y, time):
    """
    Compute flow configuration given by
    Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159
    JAX version
    """
    # Base flow parameters
    xo = 5.0
    yo = 0.0
    beta = 5.0
    gamma = 1.4
    u_base = 1.0
    v_base = 0.0

    # Account for vortex movement with time
    xmut = x - u_base * time
    ymvt = y - v_base * time

    # Calculate distance from vortex center
    r = jnp.sqrt((xmut - xo)**2 + (ymvt - yo)**2)

    # Perturbed velocity field
    u = u_base - beta * jnp.exp(1 - r**2) * (ymvt - yo) / (2 * jnp.pi)
    v = v_base + beta * jnp.exp(1 - r**2) * (xmut - xo) / (2 * jnp.pi)

    # Perturbed density and pressure
    rho1 = (1 - ((gamma - 1) * beta**2 * jnp.exp(2 * (1 - r**2)) / 
                (16 * gamma * jnp.pi**2)))**(1 / (gamma - 1))
    p1 = rho1**gamma

    Q = jnp.stack([rho1, rho1 * u, rho1 * v, p1 / (gamma - 1) + 0.5 * rho1 * (u**2 + v**2)], axis=-1)
    return Q

bc = {
    "in": isentropic_vortex_ic_2d,
    "out": isentropic_vortex_ic_2d,
    "wall": isentropic_vortex_ic_2d,
}

class Euler2D_IsentropicVortex(DGSolver):

    def __init__(self, mesh_path, order, bc):
        self.mesh = TriMesh.from_gambit(mesh_path, order)
        self.bc = bc

        super().__init__(Euler2D(1.4), local_lax_friedrichs_matlab, self.mesh, bc)

    def cut_off_filter_2d(self, Nc, frac):
        """
        Initialize 2D cut off filter matrix of order Norderin
        """
        filterdiag = np.ones(self.mesh.Np)

        # Build exponential filter
        sk = 0
        for i in range(self.mesh.order + 1):
            for j in range(self.mesh.order + 1 - i):
                if i + j >= Nc:
                    filterdiag[sk] = frac
                sk += 1

        self.Filt = self.mesh.Vand @ jnp.diag(filterdiag) @ self.mesh.invVand

    def euler_dt_2d(self, Q, gamma):
        """
        Compute the time step dt for the compressible Euler equations
        """
        # Extract conserved variables
        rho =  Q[:,:,0]
        rhou = Q[:,:,1]
        rhov = Q[:,:,2]
        Ener = Q[:,:,3]

        # Get values at boundary points
        rho =   rho.flatten(order='F')[self.mesh.vmapM]
        rhou = rhou.flatten(order='F')[self.mesh.vmapM]
        rhov = rhov.flatten(order='F')[self.mesh.vmapM]
        Ener = Ener.flatten(order='F')[self.mesh.vmapM]

        u = rhou / (jnp.abs(rho) + 1e-15)
        v = rhov / (jnp.abs(rho) + 1e-15)
        p = (gamma - 1.0) * (Ener - rho * (u**2 + v**2) / 2)
        c = jnp.sqrt(jnp.abs(gamma * p / (jnp.abs(rho) + 1e-15)))
        dt = 1.0 / jnp.max(((self.mesh.order + 1)**2) * 0.5 * self.mesh.face_scale.flatten(order='F') * (jnp.sqrt(u**2 + v**2) + c))
        return dt

    @partial(jax.jit, static_argnums=0)
    def euler_rhs_2d(self, Q, time):
        """
        Evaluate RHS in 2D Euler equations, the setup of rhs is defined in DGSolver
        local Lax-Friedrich flux_matlab
        """
        rhsQ = self.rhs(Q, time)
        rhsQ = jnp.einsum('ij, jkn -> ikn', self.Filt, rhsQ) 
        return rhsQ

    def euler_2d(self, Q, final_time):
        """
        Integrate 2D Euler equations using a 5-stage RK method
        """
        # Initialize filter
        self.cut_off_filter_2d(self.mesh.order, 0.95)

        # Compute initial timestep
        gamma = 1.4
        dt = self.euler_dt_2d(Q, gamma)
        time = 0
        tstep = 1

        # Filter initial solution
        Q = jnp.einsum('ij, jkn -> ikn', self.Filt, Q)

        sol = [Q]
        # Outer time step loop
        while time < final_time:
            # Check to see if we need to adjust for final time step
            if time + dt > final_time:
                dt = final_time - time

            Q = rk4_step(self.euler_rhs_2d, Q, time, dt)

            # Increment time and compute new timestep
            time = time + dt
            print(time)
            dt = self.euler_dt_2d(Q, gamma)
            sol.append(Q)
            tstep += 1

        return sol


if __name__ == "__main__":
    
    EulerSolver = Euler2D_IsentropicVortex("Mesh_neu/vortexA04.neu", 5, bc) # isentropic_vortex_bc_2d)

    # Compute initial condition
    Q = isentropic_vortex_ic_2d(EulerSolver.mesh.x, EulerSolver.mesh.y, EulerSolver.mesh.x, EulerSolver.mesh.y, 0)

    final_time = 1. 
    # Solve problem
    # EulerSolver.test_euler_rhs_2d(Q, final_time, isentropic_vortex_bc_2d)
    # # Solve problem
    Q = EulerSolver.euler_2d(Q, final_time)
    
    exact_Q = isentropic_vortex_ic_2d(EulerSolver.mesh.x, EulerSolver.mesh.y,EulerSolver.mesh.x, EulerSolver.mesh.y, final_time)
    
    print("The maximum absolute difference between python and exact solution:", np.max(np.abs(Q[-1] - exact_Q)))

    try:
        import scipy
        # Try loading with scipy first
        f = scipy.io.loadmat('Q.mat')
        print("Successfully loaded with scipy.io.loadmat")
    except NotImplementedError as e:
        if "Please use HDF reader" in str(e):
            print("This is a MATLAB v7.3 file (HDF5 format)")
        else:
            print("Unknown error:", e)
    except Exception as e:
        print("Error:", e)

    # Access a specific variable
    # Note: For MATLAB arrays, you may need to transpose the data
    matlab_data = np.array(f['Q'])
    
    print("The maximum absolute difference between python and matlab:", np.max(np.abs(Q[-1] - matlab_data)))