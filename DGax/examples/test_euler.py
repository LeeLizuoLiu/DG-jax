import numpy as np
from DGax.mesh.TriMesh import TriMesh
from DGax.integrators.SSPRK import rk4_step
# from DGax.equations.CompressibleEuler import euler_2d
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)  # Enable double precision
jax.config.update("jax_debug_nans", True)  # Enable NaN debugging
from functools import partial

import pdb

def isentropic_vortex_bc_2d(xin, yin, nxin, nyin, mapI, mapO, mapW, mapC, Q, time):
    """
    Impose boundary conditions on 2D Euler equations on weak form (JAX version)
    """
    # Get the exact solution at the boundary points
    Qbc = isentropic_vortex_ic_2d(xin, yin, time)
    
    # Combine all boundary maps
    mapB = jnp.concatenate([mapI, mapO, mapW])
    
    # Get array dimensions
    nx, _ = Q.shape[:2]
    
    # Convert linear indices to 2D indices (vectorized)
    cols = mapB // nx
    rows = mapB % nx
    
    # Update all boundary points at once using JAX's functional syntax
    # This is more efficient than looping and works with JIT compilation
    Q_new = jnp.array(Q)  # Ensure it's a JAX array
    Q_new = Q_new.at[rows, cols, :].set(Qbc[rows, cols, :])
    
    return Q_new

def isentropic_vortex_ic_2d(x, y, time):
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

    # Initialize and fill solution array using functional updates
    Q = jnp.zeros((x.shape[0], x.shape[1], 4))
    
    # Set conservative variables
    Q = Q.at[:,:,0].set(rho1)
    Q = Q.at[:,:,1].set(rho1 * u)
    Q = Q.at[:,:,2].set(rho1 * v)
    Q = Q.at[:,:,3].set(p1 / (gamma - 1) + 0.5 * rho1 * (u**2 + v**2))

    return Q

def euler_fluxes_2d(Q, gamma):
    """
    Evaluate primitive variables and Euler flux functions (JAX version)
    """
    # Extract conserved variables
    rho = Q[...,0]
    rhou = Q[...,1]
    rhov = Q[...,2]
    Ener = Q[...,3]
    
    # Compute primitive variables
    u = rhou / (jnp.abs(rho) + 1e-15)
    v = rhov / (jnp.abs(rho) + 1e-15)
    p = (gamma - 1) * (Ener - 0.5 * (rhou * u + rhov * v))
    
    # Compute flux functions using functional syntax
    F = jnp.zeros_like(Q)
    F = F.at[...,0].set(rhou)
    F = F.at[...,1].set(rhou * u + p)
    F = F.at[...,2].set(rhov * u)
    F = F.at[...,3].set(u * (Ener + p))
    
    G = jnp.zeros_like(Q)
    G = G.at[...,0].set(rhov)
    G = G.at[...,1].set(rhou * v)
    G = G.at[...,2].set(rhov * v + p)
    G = G.at[...,3].set(v * (Ener + p))
    
    return F, G, rho, u, v, p

class Euler_2D:

    def __init__(self, mesh_path, order, bc):
        self.mesh = TriMesh.from_gambit(mesh_path, order)
        self.bc = bc

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
        Evaluate RHS in 2D Euler equations, discretized on weak form
        with a local Lax-Friedrich flux
        """
        mapM = self.mesh.mapM
        mapP = self.mesh.mapP

        # 1. Compute volume contributions
        gamma = 1.4
        F, G, rho, u, v, p = euler_fluxes_2d(Q, gamma)
        
        flux = jnp.stack((F, G), axis=-1)
        # Compute weak derivatives
        weak_flux = jnp.einsum("jknm,pjd->pknmd", flux, self.mesh.Dw)
        volume_integral = jnp.einsum("pknmd, pkmd->pkn", weak_flux, self.mesh.rs_xy)

        # 2. Compute surface contributions
        # 2.1 Evaluate '-' and '+' traces of conservative variables
        Q_flat = Q.reshape(-1, 4, order='F')
        QM = Q_flat[mapM, :]
        oQP = Q_flat[mapP, :]
        # 2.2 Set boundary conditions by modifying positive traces
        QP = self.bc(self.mesh.Fx, self.mesh.Fy, self.mesh.face_normals[...,0], self.mesh.face_normals[...,1], 
                          self.mesh.bc_maps['in'], self.mesh.bc_maps['out'], self.mesh.bc_maps['wall'], self.mesh.bc_maps['cylinder'], oQP, time)

        QM = QM.reshape(self.mesh.Nfp, self.mesh.Nfaces, self.mesh.K, QM.shape[-1], order='F') # shape = [Nfp, Nfaces, K, n_vars]
        QP = QP.reshape(self.mesh.Nfp, self.mesh.Nfaces, self.mesh.K, QP.shape[-1], order='F') # shape = [Nfp, Nfaces, K, n_vars]

        surface_integral = self.surface_integral(QM, QP, gamma)
        rhsQ = volume_integral + surface_integral
        rhsQ = jnp.einsum('ij, jkn -> ikn', self.Filt, rhsQ) 
        return rhsQ

    def lax_friedrichs_flux(self, QM, QP, face_normals, gamma):
        # QM, QP shape = [Nfp, n_vars]
        # 2.3 Evaluate primitive variables & flux functions at '-' and '+' traces
        lambda_max = self.max_local_speed(QM, QP, gamma) # lambda_max shape = [1]
        # 2.5 Lift fluxes
        fP, gP, _, _, _, _ = euler_fluxes_2d(QM, gamma) # fM, gM shape = [Nfp, n_vars]
        fM, gM, _, _, _, _ = euler_fluxes_2d(QP, gamma) # fP, gP shape = [Nfp, n_vars]

        face_flux = jnp.stack((fP + fM, gP + gM), axis=-1)
        diffusion_term =  lambda_max*(QM - QP)
        face_integral = jnp.einsum('pd, pvd -> pv', face_normals, face_flux) + diffusion_term # .reshape(self.mesh.Nfp*self.mesh.Nfaces, self.mesh.K, QP.shape[-1], order='F')
        return face_integral

    def surface_integral(self, QM, QP, gamma):
        face_integral = jax.vmap(jax.vmap(self.lax_friedrichs_flux ,in_axes=(1,1,1,None), out_axes=1),in_axes=(1,1,1,None), out_axes=1)(QM, QP, self.mesh.face_normals, gamma)
        return - jnp.einsum('inf,nfkl->ikl',self.mesh.Lift, self.mesh.face_scale/2 * face_integral)

    def max_local_speed(self, QM, QP, gamma):
        _, _, rhoM, uM, vM, pM = euler_fluxes_2d(QM, gamma) # rhoM, uM, vM, pM shape = [Nfp]
        _, _, rhoP, uP, vP, pP = euler_fluxes_2d(QP, gamma) # rhoP, uP, vP, pP shape = [Nfp]
        lambda_val = jnp.maximum(
            jnp.sqrt(uM**2 + vM**2) + jnp.sqrt(jnp.abs(gamma * pM / (jnp.abs(rhoM)+1e-15))),
            jnp.sqrt(uP**2 + vP**2) + jnp.sqrt(jnp.abs(gamma * pP / (jnp.abs(rhoP)+1e-15)))
        ) # lambda_val shape = [Nfp]
        lambda_max = jnp.max(lambda_val) # lambda_max shape = [1]
        return lambda_max

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
    
    EulerSolver = Euler_2D("Mesh_neu/vortexA04.neu", 5, isentropic_vortex_bc_2d)

    # Compute initial condition
    Q = isentropic_vortex_ic_2d(EulerSolver.mesh.x, EulerSolver.mesh.y, 0)

    final_time = 0.02 
    # Solve problem
    # EulerSolver.test_euler_rhs_2d(Q, final_time, isentropic_vortex_bc_2d)
    # # Solve problem
    Q = EulerSolver.euler_2d(Q, final_time)
    
    exact_Q = isentropic_vortex_ic_2d(EulerSolver.mesh.x, EulerSolver.mesh.y, final_time)
    
    print("The maximum absolute difference between python and matlab:", np.max(np.abs(Q[-1] - exact_Q)))