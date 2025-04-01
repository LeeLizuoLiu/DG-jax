import numpy as np

def isentropic_vortex_bc_2d(xin, yin, nxin, nyin, mapI, mapO, mapW, mapC, Q, time):
    """
    Impose boundary conditions on 2D Euler equations on weak form
    """
    # Get the exact solution at the boundary points
    Qbc = isentropic_vortex_ic_2d(xin, yin, time)

    # Combine all boundary maps
    mapB = np.concatenate([mapI, mapO, mapW])

    # Apply boundary conditions for each component
    for n in range(4):
        Qn = Q[:,:,n].copy()
        Qbcn = Qbc[:,:,n]
        Qn.flatten(order='F')[mapB] = Qbcn.flatten(order='F')[mapB]
        Q[:,:,n] = Qn

    return Q

def isentropic_vortex_ic_2d(x, y, time):
    """
    Compute flow configuration given by
    Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159
    """
    # Base flow parameters
    xo = 5
    yo = 0
    beta = 5
    gamma = 1.4
    rho = 1
    u_base = 1
    v_base = 0
    p = 1

    # Account for vortex movement with time
    xmut = x - u_base * time
    ymvt = y - v_base * time

    # Calculate distance from vortex center
    r = np.sqrt((xmut - xo)**2 + (ymvt - yo)**2)

    # Perturbed velocity field
    u = u_base - beta * np.exp(1 - r**2) * (ymvt - yo) / (2 * np.pi)
    v = v_base + beta * np.exp(1 - r**2) * (xmut - xo) / (2 * np.pi)

    # Perturbed density and pressure
    rho1 = (1 - ((gamma - 1) * beta**2 * np.exp(2 * (1 - r**2)) / 
                (16 * gamma * np.pi**2)))**(1 / (gamma - 1))
    p1 = rho1**gamma

    # Initialize solution array
    Q = np.zeros((x.shape[0], x.shape[1], 4))

    # Set conservative variables
    Q[:,:,0] = rho1
    Q[:,:,1] = rho1 * u
    Q[:,:,2] = rho1 * v
    Q[:,:,3] = p1 / (gamma - 1) + 0.5 * rho1 * (u**2 + v**2)

    return Q