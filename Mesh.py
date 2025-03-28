import numpy as np
import jax.numpy as jnp
import math

import numpy as np
from scipy.special import gamma

class Elements:
    @staticmethod
    def JacobiP(x, alpha, beta, N):
        """
        Evaluate Jacobi Polynomial of type (alpha,beta) > -1
        
        Parameters:
        x: input points
        alpha, beta: polynomial parameters
        N: polynomial order
        """
        x = np.atleast_1d(x)
        
        # Ensure x is a row vector
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        PL = np.zeros((N+1, x.shape[1]))
        
        # Initial values P_0(x) and P_1(x)
        gamma0 = (2**(alpha+beta+1)) / (alpha+beta+1) * \
                 gamma(alpha+1) * gamma(beta+1) / gamma(alpha+beta+1)
        PL[0, :] = 1.0 / np.sqrt(gamma0)
        
        if N == 0:
            return PL[0, :].reshape(-1, 1)
        
        gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3) * gamma0
        PL[1, :] = ((alpha+beta+2)*x/2 + (alpha-beta)/2) / np.sqrt(gamma1)
        
        if N == 1:
            return PL[1, :].reshape(-1, 1)
        
        # Repeat value in recurrence
        aold = 2/(2+alpha+beta) * np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
        
        # Forward recurrence
        for i in range(N-1):
            h1 = 2*i + alpha + beta
            anew = 2/(h1+2) * np.sqrt((i+1)*(i+1+alpha+beta)*(i+1+alpha)*(i+1+beta) / (h1+1)/(h1+3))
            bnew = -(alpha**2 - beta**2) / h1 / (h1+2)
            
            PL[i+2, :] = 1/anew * (-aold*PL[i, :] + (x-bnew)*PL[i+1, :])
            aold = anew
        
        return PL[N, :].reshape(-1, 1)

    @staticmethod
    def JacobiGL(alpha, beta, N):
        """Compute Gauss-Lobatto quadrature points"""
        if N == 1:
            return np.array([-1.0, 1.0])
        
        # For now, using a placeholder. In a real implementation, 
        # you'd want to use JacobiGQ
        xint = np.linspace(-1, 1, N-1)
        return np.concatenate([[-1], xint, [1]])

    @staticmethod
    def Vandermonde1D(N, r):
        """Initialize the 1D Vandermonde Matrix"""
        r = np.atleast_1d(r)
        V1D = np.zeros((len(r), N+1))
        
        for j in range(N+1):
            V1D[:, j] = Elements.JacobiP(r, 0, 0, j).flatten()
        
        return V1D

    @staticmethod
    def Warpfactor(N, rout):
        """
        Compute scaled warp function at order N based on rout interpolation nodes
        
        Parameters:
        -----------
        N : int
            Order of approximation
        rout : array-like
            Interpolation nodes
        
        Returns:
        --------
        warp : ndarray
            Warp factor
        """
        # Compute LGL and equidistant node distribution
        LGLr = Elements.JacobiGL(0, 0, N)
        req = np.linspace(-1, 1, N+1)
        
        # Compute V based on req
        Veq = Elements.Vandermonde1D(N, req)
        
        # Evaluate Lagrange polynomial at rout
        rout = np.asarray(rout)
        Pmat = np.column_stack([Elements.JacobiP(rout, 0, 0, i).flatten() for i in range(N+1)])
        
        # Solve for Lagrange interpolation matrix
        Lmat = np.linalg.solve(Veq.T, Pmat.T).T
        
        # Compute warp factor
        warp = Lmat @ (LGLr - req)
        
        # Scale factor
        zerof = np.abs(rout) < 1.0 - 1.0e-10
        sf = 1.0 - (zerof * rout)**2
        
        # Final warp computation
        warp = warp / sf + warp * (zerof - 1)
        
        return warp

    @staticmethod
    def get_nodes_and_warp(N, node_type='LGL'):
        """
        Convenience method to get nodes and their warp factor
        
        Parameters:
        -----------
        N : int
            Order of approximation
        node_type : str, optional
            Type of nodes (default is 'LGL' - Legendre-Gauss-Lobatto)
        
        Returns:
        --------
        nodes : ndarray
            Interpolation nodes
        warp : ndarray
            Warp factor for the nodes
        """
        if node_type == 'LGL':
            nodes = Elements.JacobiGL(0, 0, N)
            warp = Elements.Warpfactor(N, nodes)
            return nodes, warp
        else:
            raise ValueError(f"Unsupported node type: {node_type}")

    @staticmethod
    def rstoab(r, s):
        """
        Convert a 2D mesh in r-s coordinates to a 2D mesh in a-b coordinates.
        """
        a = np.where(s != 1, 2 * (1 + r) / (1 - s) - 1, -1)
        b = s
        return a, b
    
    @staticmethod
    def Nodes2D(N):
        """
        Compute (x,y) nodes in equilateral triangle for polynomial of order N
        
        Parameters:
        -----------
        N : int
            Order of polynomial approximation
        
        Returns:
        --------
        x, y : ndarray
            Coordinates of nodes in the equilateral triangle
        """
        # Optimized alpha parameters for different orders
        alpopt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 
                  1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]
        
        # Set optimized parameter, alpha, depending on order N
        if N < 16:
            alpha = alpopt[N]
        else:
            alpha = 5/3
        
        # Total number of nodes
        Np = (N+1)*(N+2)//2
        
        # Create equidistributed nodes on equilateral triangle
        L1 = np.zeros(Np)
        L2 = np.zeros(Np)
        L3 = np.zeros(Np)
        
        sk = 0
        for n in range(1, N+2):
            for m in range(1, N+3-n):
                L1[sk] = (n-1)/N
                L3[sk] = (m-1)/N
                sk += 1
        
        L2 = 1.0 - L1 - L3
        
        # Convert to Cartesian coordinates
        x = -L2 + L3
        y = (-L2 - L3 + 2*L1) / math.sqrt(3.0)
        
        # Compute blending function at each node for each edge
        blend1 = 4 * L2 * L3
        blend2 = 4 * L1 * L3
        blend3 = 4 * L1 * L2
        
        # Amount of warp for each node, for each edge
        warpf1 =Elements.Warpfactor(N, L3 - L2)
        warpf2 =Elements.Warpfactor(N, L1 - L3)
        warpf3 =Elements.Warpfactor(N, L2 - L1)
        
        # Combine blend & warp
        warp1 = blend1 * warpf1 * (1 + (alpha*L1)**2)
        warp2 = blend2 * warpf2 * (1 + (alpha*L2)**2)
        warp3 = blend3 * warpf3 * (1 + (alpha*L3)**2)
        
        # Accumulate deformations associated with each edge
        x = x + 1*warp1 + math.cos(2*math.pi/3)*warp2 + math.cos(4*math.pi/3)*warp3
        y = y + 0*warp1 + math.sin(2*math.pi/3)*warp2 + math.sin(4*math.pi/3)*warp3
        
        return x, y    
    
    @staticmethod
    def xytors(x, y):
        """
        Convert (x,y) coordinates in equilateral triangle to (r,s) coordinates in standard triangle

        Parameters:
        -----------
        x : array-like
            x-coordinates in equilateral triangle
        y : array-like
            y-coordinates in equilateral triangle

        Returns:
        --------
        r, s : tuple of ndarrays
            Coordinates in standard triangle
        """
        # Ensure inputs are NumPy arrays
        x = np.asarray(x)
        y = np.asarray(y)

        # Compute barycentric coordinates
        L1 = (np.sqrt(3.0)*y + 1.0) / 3.0
        L2 = (-3.0*x - np.sqrt(3.0)*y + 2.0) / 6.0
        L3 = (3.0*x - np.sqrt(3.0)*y + 2.0) / 6.0

        # Convert to (r,s) coordinates
        r = -L2 + L3 - L1
        s = -L2 - L3 + L1

        return r, s
    
    @staticmethod
    def Simplex2DP(a, b, i, j):
        """
        Evaluate 2D orthonormal polynomial on simplex at (a,b) of order (i,j)

        Parameters:
        -----------
        a : array-like
            First coordinate
        b : array-like
            Second coordinate
        i : int
            First polynomial order
        j : int
            Second polynomial order

        Returns:
        --------
        P : ndarray
            Evaluated polynomial values
        """
        a, b = np.asarray(a), np.asarray(b)

        # Compute Jacobi polynomials
        h1 = Elements.JacobiP(a, 0, 0, i)
        h2 = Elements.JacobiP(b, 2*i+1, 0, j)

        # Compute polynomial values
        P = np.sqrt(2.0) * h1 * h2 * (1-b)**i

        return P

    @staticmethod
    def Vandermonde2D(N, r, s):
        """
        Initialize the 2D Vandermonde Matrix

        Parameters:
        -----------
        N : int
            Polynomial order
        r : array-like
            r-coordinates
        s : array-like
            s-coordinates

        Returns:
        --------
        V2D : ndarray
            2D Vandermonde matrix
        """
        r, s = np.asarray(r), np.asarray(s)

        # Total number of basis functions
        Np = (N+1)*(N+2)//2

        # Initialize Vandermonde matrix
        V2D = np.zeros((len(r), Np))

        # Transfer to (a,b) coordinates
        a, b = Elements.rstoab(r, s)

        # Build the Vandermonde matrix
        sk = 0
        for i in range(N+1):
            for j in range(N+1-i):
                V2D[:, sk] = Elements.Simplex2DP(a, b, i, j)
                sk += 1

        return V2D
    @staticmethod
    def Grad2D(u, rx, sx, ry, sy, Dr, Ds):
        """
        Compute 2D gradient field of scalar u
        
        Parameters:
        -----------
        u : ndarray
            Scalar field
        rx, sx, ry, sy : ndarray
            Metric terms
        Dr, Ds : ndarray
            Differentiation matrices
        
        Returns:
        --------
        ux, uy : tuple of ndarrays
            Gradient components
        """
        ur = Dr @ u
        us = Ds @ u
        ux = rx * ur + sx * us
        uy = ry * ur + sy * us
        
        return ux, uy

    @staticmethod
    def Div2D(u, v, rx, sx, ry, sy, Dr, Ds):
        """
        Compute the 2D divergence of the vectorfield (u,v)
        
        Parameters:
        -----------
        u, v : ndarray
            Vector field components
        rx, sx, ry, sy : ndarray
            Metric terms
        Dr, Ds : ndarray
            Differentiation matrices
        
        Returns:
        --------
        divu : ndarray
            Divergence of the vector field
        """
        ur = Dr @ u
        us = Ds @ u
        vr = Dr @ v
        vs = Ds @ v
        
        divu = rx * ur + sx * us + ry * vr + sy * vs
        
        return divu

    @staticmethod
    def Curl2D(ux, uy, uz=None, rx=None, sx=None, ry=None, sy=None, Dr=None, Ds=None):
        """
        Compute 2D curl-operator in (x,y) plane
        
        Parameters:
        -----------
        ux, uy : ndarray
            x and y components of vector field
        uz : ndarray, optional
            z component of vector field
        rx, sx, ry, sy : ndarray, optional
            Metric terms
        Dr, Ds : ndarray, optional
            Differentiation matrices
        
        Returns:
        --------
        vx, vy, vz : tuple of ndarrays
            Curl components
        """
        uxr = Dr @ ux if Dr is not None else np.gradient(ux)
        uxs = Ds @ ux if Ds is not None else np.gradient(ux)
        uyr = Dr @ uy if Dr is not None else np.gradient(uy)
        uys = Ds @ uy if Ds is not None else np.gradient(uy)
        
        # Compute vz (2D curl in z-direction)
        if rx is not None and sx is not None and ry is not None and sy is not None:
            vz = rx * uyr + sx * uys - ry * uxr - sy * uxs
        else:
            vz = uyr - uxs
        
        # If uz is provided, compute additional curl components
        vx, vy = None, None
        if uz is not None:
            uzr = Dr @ uz if Dr is not None else np.gradient(uz)
            uzs = Ds @ uz if Ds is not None else np.gradient(uz)
            
            if rx is not None and sx is not None and ry is not None and sy is not None:
                vx = ry * uzr + sy * uzs
                vy = -rx * uzr - sx * uzs
            else:
                vx = uzs
                vy = -uzr
        
        return vx, vy, vz

    @staticmethod
    def Filter2D(Norder, Nc, V, sp=1):
        """
        Initialize 2D filter matrix of order sp and cutoff Nc
        
        Parameters:
        -----------
        Norder : int
            Polynomial order
        Nc : int
            Cutoff frequency
        sp : int, optional
            Filter strength (default 1)
        
        Returns:
        --------
        F : ndarray
            Filter matrix
        """
        # Total number of basis functions
        Np = (Norder+1)*(Norder+2)//2
        
        # Initialize filter diagonal
        filterdiag = np.ones(Np)
        
        # Compute alpha
        alpha = -np.log(np.finfo(float).eps)
        
        # Build exponential filter
        sk = 0
        for i in range(Norder+1):
            for j in range(Norder+1-i):
                if i+j >= Nc:
                    filterdiag[sk] = np.exp(-alpha * ((i+j - Nc)/(Norder-Nc))**sp)
                sk += 1
        
        F = V @ np.diag(filterdiag) @ np.linalg.inv(V)
        
        return F

    @staticmethod
    def Dmatrices2D(N, r, s, V):
        """
        Initialize the (r,s) differentiation matrices on the simplex
        
        Parameters:
        -----------
        N : int
            Polynomial order
        r, s : ndarray
            Coordinate points
        V : ndarray
            Vandermonde matrix
        
        Returns:
        --------
        Dr, Ds : tuple of ndarrays
            Differentiation matrices
        """
        # Get gradient of Vandermonde matrix
        Vr, Vs = Elements.GradVandermonde2D(N, r, s)
        
        # Compute differentiation matrices
        Dr = Vr @ np.linalg.inv(V)
        Ds = Vs @ np.linalg.inv(V)
        
        return Dr, Ds

    @staticmethod
    def GradVandermonde2D(N, r, s):
        """
        Initialize the gradient of the modal basis at (r,s)
        
        Parameters:
        -----------
        N : int
            Polynomial order
        r, s : ndarray
            Coordinate points
        
        Returns:
        --------
        V2Dr, V2Ds : tuple of ndarrays
            Gradient of Vandermonde matrix
        """
        # Total number of basis functions
        Np = (N+1)*(N+2)//2
        
        # Initialize gradient matrices
        V2Dr = np.zeros((len(r), Np))
        V2Ds = np.zeros((len(r), Np))
        
        # Convert to tensor-product coordinates
        a, b = Elements.rstoab(r, s)
        
        # Compute gradient
        sk = 0
        for i in range(N+1):
            for j in range(N+1-i):
                V2Dr[:, sk], V2Ds[:, sk] = Elements.GradSimplex2DP(a, b, i, j)
                sk += 1
        
        return V2Dr, V2Ds

    @staticmethod
    def GradSimplex2DP(a, b, id, jd):
        """
        Return the derivatives of the modal basis on the 2D simplex
        
        Parameters:
        -----------
        a, b : ndarray
            Coordinates
        id, jd : int
            Polynomial orders
        
        Returns:
        --------
        dmodedr, dmodeds : tuple of ndarrays
            Derivatives in r and s directions
        """
        # Compute Jacobi polynomials and their derivatives
        fa = Elements.JacobiP(a, 0, 0, id)
        dfa = Elements.GradJacobiP(a, 0, 0, id)
        gb = Elements.JacobiP(b, 2*id+1, 0, jd)
        dgb = Elements.GradJacobiP(b, 2*id+1, 0, jd)
        
        # r-derivative
        dmodedr = dfa * gb
        if id > 0:
            dmodedr *= (0.5 * (1-b))**(id-1)
        
        # s-derivative
        dmodeds = dfa * (gb * (0.5 * (1+a)))
        if id > 0:
            dmodeds *= (0.5 * (1-b))**(id-1)
        
        tmp = dgb * ((0.5 * (1-b))**id)
        if id > 0:
            tmp -= 0.5 * id * gb * ((0.5 * (1-b))**(id-1))
        
        dmodeds += fa * tmp
        
        # Normalize
        dmodedr *= 2**(id+0.5)
        dmodeds *= 2**(id+0.5)
        
        return dmodedr, dmodeds

    @staticmethod
    def GradJacobiP(x, alpha, beta, N):
        """
        Compute gradient of Jacobi polynomials
        
        Parameters:
        -----------
        x : ndarray
            Input points
        alpha, beta : float
            Jacobi polynomial parameters
        N : int
            Polynomial order
        
        Returns:
        --------
        dP : ndarray
            Gradient of Jacobi polynomials
        """
        # Handle zero order case
        if N == 0:
            return np.zeros_like(x)
        
        # Compute gradient
        dP = np.sqrt(N * (N + alpha + beta + 1)) * \
             Elements.JacobiP(x, alpha+1, beta+1, N-1)
        
        return dP

    @staticmethod
    def Lift2D(N, Np, Nfaces, Nfp, Fmask, r, V):
        """
        Compute surface to volume lift term for DG formulation
        
        Parameters:
        -----------
        N : int
            Polynomial order
        Np : int
            Number of points
        Nfaces : int
            Number of faces
        Nfp : int
            Number of points per face
        Fmask : ndarray
            Face mask indices
        r : ndarray
            Reference coordinates
        V : ndarray
            Vandermonde matrix
        
        Returns:
        --------
        LIFT : ndarray
            Lift matrix
        """
        # Initialize element matrix
        Emat = np.zeros((Np, Nfaces * Nfp))
        
        # Process each face
        for face in range(Nfaces):
            # Extract face coordinates
            faceR = r[Fmask[:, face]]
            
            # Compute 1D Vandermonde matrix for the face
            V1D = Elements.Vandermonde1D(N, faceR)
            
            # Compute mass matrix for the edge
            massEdge = np.linalg.inv(V1D @ V1D.T)
            
            # Populate Emat for this face
            Emat[Fmask[:, face], face*Nfp:(face+1)*Nfp] = massEdge
        
        # Compute LIFT matrix
        LIFT = V @ (V.T @ Emat)
        
        return LIFT