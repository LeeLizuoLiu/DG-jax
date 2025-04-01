import numpy as np
from Mesh import Elements
from constants import *
from euler_BC_IC import isentropic_vortex_bc_2d, isentropic_vortex_ic_2d


def euler_fluxes_2d(Q, gamma):
    """
    Evaluate primitive variables and Euler flux functions
    """
    # Extract conserved variables
    rho =  Q[...,0]
    rhou = Q[...,1]
    rhov = Q[...,2]
    Ener = Q[...,3]
    
    # Compute primitive variables
    u = rhou / rho
    v = rhov / rho
    p = (gamma - 1) * (Ener - 0.5 * (rhou * u + rhov * v))
    
    # Compute flux functions
    F = np.zeros_like(Q)
    F[...,0] = rhou
    F[...,1] = rhou * u + p
    F[...,2] = rhov * u
    F[...,3] = u * (Ener + p)
    
    G = np.zeros_like(Q)
    G[...,0] = rhov
    G[...,1] = rhou * v
    G[...,2] = rhov * v + p
    G[...,3] = v * (Ener + p)
    
    return F, G, rho, u, v, p

class Euler_2D(Elements):

    def __init__(self, mesh_path, order):
        super().__init__(mesh_path, order)

    def cut_off_filter_2d(self, Nc, frac):
        """
        Initialize 2D cut off filter matrix of order Norderin
        """
        filterdiag = np.ones(self.Np)

        # Build exponential filter
        sk = 0
        for i in range(self.N + 1):
            for j in range(self.N + 1 - i):
                if i + j >= Nc:
                    filterdiag[sk] = frac
                sk += 1

        F = self.V @ np.diag(filterdiag) @ self.invV
        return F

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
        rho =   rho.flatten('F')[self.vmapM]
        rhou = rhou.flatten('F')[self.vmapM]
        rhov = rhov.flatten('F')[self.vmapM]
        Ener = Ener.flatten('F')[self.vmapM]

        u = rhou / rho
        v = rhov / rho
        p = (gamma - 1.0) * (Ener - rho * (u**2 + v**2) / 2)
        c = np.sqrt(np.abs(gamma * p / rho))

        dt = 1.0 / np.max(((self.N + 1)**2) * 0.5 * self.Fscale.flatten() * (np.sqrt(u**2 + v**2) + c))

        return dt

    def euler_rhs_2d(self, Q, time, exact_solution_bc):
        """
        Evaluate RHS in 2D Euler equations, discretized on weak form
        with a local Lax-Friedrich flux
        """
        vmapM = self.vmapM.reshape(self.Nfp * self.Nfaces, self.K, order='F')
        vmapP = self.vmapP.reshape(self.Nfp * self.Nfaces, self.K, order='F')

        # 1. Compute volume contributions
        gamma = 1.4
        F, G, rho, u, v, p = euler_fluxes_2d(Q, gamma)

        # Compute weak derivatives
        rhsQ = np.zeros_like(Q)
        for n in range(4):
            dFdr = self.Drw @ F[:,:,n]
            dFds = self.Dsw @ F[:,:,n]
            dGdr = self.Drw @ G[:,:,n]
            dGds = self.Dsw @ G[:,:,n]
            rhsQ[:,:,n] = (self.rx * dFdr + self.sx * dFds) + (self.ry * dGdr + self.sy * dGds)

        # 2. Compute surface contributions
        # 2.1 Evaluate '-' and '+' traces of conservative variables
        QM = np.zeros((self.Nfp * self.Nfaces, self.K, 4))
        QP = np.zeros((self.Nfp * self.Nfaces, self.K, 4))

        for n in range(4):
            Qn = Q[:,:,n].flatten('F')
            QM[:,:,n] = Qn[vmapM]
            QP[:,:,n] = Qn[vmapP]

        # 2.2 Set boundary conditions by modifying positive traces
        if exact_solution_bc is not None:
            QP = exact_solution_bc(self.Fx, self.Fy, self.nx, self.ny, 
                                  self.bc_maps['mapI'], self.bc_maps['mapO'], self.bc_maps['mapW'], self.bc_maps['mapC'], QP, time)

        # 2.3 Evaluate primitive variables & flux functions at '-' and '+' traces
        fM, gM, rhoM, uM, vM, pM = euler_fluxes_2d(QM, gamma)
        fP, gP, rhoP, uP, vP, pP = euler_fluxes_2d(QP, gamma)

        # 2.4 Compute local Lax-Friedrichs/Rusanov numerical fluxes
        lambda_val = np.maximum(
            np.sqrt(uM**2 + vM**2) + np.sqrt(np.abs(gamma * pM / rhoM)),
            np.sqrt(uP**2 + vP**2) + np.sqrt(np.abs(gamma * pP / rhoP))
        )

        lambda_val = lambda_val.reshape(self.Nfp, self.Nfaces * self.K, order='F')
        lambda_max = np.max(lambda_val, axis=0)
        lambda_val = np.ones((self.Nfp, 1)) @ lambda_max.reshape(1, -1)
        lambda_val = lambda_val.reshape(self.Nfp * self.Nfaces, self.K, order='F')

        # 2.5 Lift fluxes
        for n in range(4):
            nflux = self.nx * (fP[:,:,n] + fM[:,:,n]) + self.ny * (gP[:,:,n] + gM[:,:,n]) + \
                    lambda_val * (QM[:,:,n] - QP[:,:,n])
            rhsQ[:,:,n] = rhsQ[:,:,n] - self.lift @ (self.Fscale * nflux / 2)

        return rhsQ

    def euler_2d(self, Q, final_time, bc):
        """
        Integrate 2D Euler equations using a 5-stage RK method
        """
        # Initialize filter
        Filt = self.cut_off_filter_2d(self.N, 0.95)

        # Compute initial timestep
        gamma = 1.4
        dt = self.euler_dt_2d(Q, gamma)
        time = 0
        tstep = 1

        # Storage for low storage RK time stepping
        rhsQ = np.zeros_like(Q)
        resQ = np.zeros_like(Q)

        # Filter initial solution
        for n in range(4):
            Q[:,:,n] = Filt @ Q[:,:,n]

        sol = [Q]
        # Outer time step loop
        while time < final_time:
            # Check to see if we need to adjust for final time step
            if time + dt > final_time:
                dt = final_time - time

            for INTRK in range(5):
                # Compute right hand side of compressible Euler equations
                rhsQ = self.euler_rhs_2d(Q, time, bc)

                # Filter residual
                for n in range(4):
                    rhsQ[:,:,n] = Filt @ rhsQ[:,:,n]

                # Initiate and increment Runge-Kutta residuals
                resQ = rk4a[INTRK] * resQ + dt * rhsQ

                # Update fields
                Q = Q + rk4b[INTRK] * resQ

            # Increment time and compute new timestep
            time = time + dt
            print(f"Time: {time}")
            dt = self.euler_dt_2d(Q, gamma)
            sol.append(Q)
            tstep += 1

        return sol


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    EulerSolver = Euler_2D("vortexA04.neu", 3)

    # Compute initial condition
    Q = isentropic_vortex_ic_2d(EulerSolver.x, EulerSolver.y, 0)

    final_time = 1.0
    # Solve problem
    Q = EulerSolver.euler_2d(Q, final_time, isentropic_vortex_bc_2d)
