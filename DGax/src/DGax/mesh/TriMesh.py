# src/DGax/mesh/TriMesh.py
"""Triangular mesh Management"""
from dataclasses import dataclass
from typing import Tuple, List, Dict
import jax.numpy as jnp
from jax import Array
import numpy as np
from recursivenodes.nodes import warburton
from recursivenodes.polynomials import proriolkoornwinderdubinervandermondegrad as VandGrad
from recursivenodes.polynomials import proriolkoornwinderdubinervandermonde as Vandermonde
from .mesh import Mesh
from .TriElement import TriElements
from .io import read_gambit_neu
from .generator_gmsh import generate_periodic_square_mesh
import pdb

@dataclass(frozen=True)
class TriMesh(Mesh):
    """
    Complete triangular mesh (geometry + connectivity)
    All arrays are JAX arrays and immutable
    """
    # --- Geometry data
    x: Array                     # [Np, K] - Physical x-coordinates
    y: Array                     # [Np, K] - Physical y-coordinates
    Fx: Array                    # [Nfp*Nfaces, K] - Face x-coordinates
    Fy: Array                    # [Nfp*Nfaces, K] - Face y-coordinates
    Vand: Array                  # [Np, Np] - Vandermonde matrix
    invVand: Array               # [Np, Np] - Inverse Vandermonde matrix
    J: Array                   # [Np, K] - Volume Jacobian
    Dw: Array                    # Weak differentiation matrix [dr, ds]
    rs_xy: Array                 # [dr/dx, dr/dy]
                                 # [ds/dx, ds/dy]
    face_normals: Array          # [Nfp*Nfaces, K, 2] - Face normals (nx, ny)
    face_sJ: Array             # [Nfp, Nfaces, K] - Surface Jacobian
    face_scale: Array            # [Nfp, Nfaces, K] - Fscale = sJ / J
    Lift: Array                  # [Np, Nfaces*Nfp] - Surface to volume lift operator
    
    # --- Connectivity data
    # EToV: Array                # [K, 3] - Element-to-vertex connectivity
    EToE: Array                  # [K, Nfaces] - Element-to-element
    EToF: Array                  # [K, Nfaces] - Element-to-face
    mapM: Array                  # [Nfp*Nfaces, K] - Volume node indices
    mapP: Array                  # [Nfp*Nfaces, K] - Neighbor volume node indices
    vmapM: Array                 # [Nfp, Nfaces, K] - Volume node indices
    vmapP: Array                 # [Nfp, Nfaces, K] - Neighbor volume node indices
    bc_maps: Dict[str, Array]    # {"wall": indices, "inlet": indices, ...}
    
    # --- Metadata
    order: int                   # Polynomial order
    Np: int                      # Nodes per element
    Nfp: int                     # Nodes per face
    K: int                       # Number of elements
    Nfaces: int = 3              # Fixed for triangles
    
    # --- Properties
    @property
    def ndim(self) -> int:
        return 2
    
    @property
    def n_elements(self) -> int:
        return self.K
    
    @property
    def n_nodes_per_element(self) -> int:
        return self.Np

    @property
    def mass_matrix(self) -> Array:
        """Compute the global mass matrix for integration."""
        # Reference mass matrix: M_ref = Vand @ Vand.T
        M_ref = self.Vand @ self.Vand.T  # [Np, Np]
        
        # For each element, mass matrix = J_k * M_ref
        # J has shape [Np, K], we need to handle element-wise
        mass_matrices = jnp.einsum('ij,jk->ijk', M_ref, self.J)  # [Np, Np, K]
        
        return mass_matrices
    
    def integrate_over_domain(self, u: Array) -> Array:
        """
        Highly optimized integration using mass matrix property.
    
        For nodal DG with orthonormal basis, the mass matrix is diagonal,
        but for general basis, we need the full matrix.
    
        Args:
            u: Array of shape [Np, K, n_vars] or [Np, K]
    
        Returns:
            Integral value (scalar for each variable)
        """
        if u.ndim == 2:
            u = u[..., jnp.newaxis]
    
        Np, K, n_vars = u.shape
    
        # Precompute: (1^T M_ref) where 1 is vector of ones
        ones = jnp.ones((Np, 1))
        ones_T_M_ref = ones.T @ self.Vand @ self.Vand.T  # [1, Np]
    
        # For each element: ∫ u dΩ_k = (1^T M_ref) * (J[:, k] * u[:, k, :])
        # Sum over nodes: sum_i (ones_T_M_ref[0,i] * J[i,k] * u[i,k,v])
        # Vectorized: [1, Np] * [Np, K, n_vars] * [Np, K, 1] -> sum over Np
        # Alternative using einsum (often faster):
        integrals_per_element = jnp.einsum('i,ik,ikv->kv', ones_T_M_ref[0], self.J, u)
    
        total_integral = jnp.sum(integrals_per_element, axis=0)  # [n_vars]
    
        return total_integral
    
    def get_boundary_mask(self, bc_type: str) -> Array:
        """Get linear indices for boundary type"""
        return self.bc_maps.get(bc_type, jnp.array([], dtype=int))
    
    @classmethod
    def from_gambit(
        cls, 
        mesh_path: str, 
        order: int,
        NODETOL: float = 1e-10
    ) -> "TriMesh":
        """Factory: read file + compute geometry and connectivity (one-time cost)"""
        # Step 1: Read raw mesh data (NumPy)
        VX, VY, EToV, BCType = read_gambit_neu(mesh_path)
        
        # Step 2: Build mesh
        return cls.build(
            VX=VX,
            VY=VY,
            EToV=EToV,
            BCType=BCType,
            order=order,
            NODETOL=NODETOL
        )   
        
    @classmethod
    def from_gmsh_periodic(
        cls,
        order: int,
        h: float = 0.1,
        domain_size: float = 1.0,
        NODETOL: float = 1e-10
    ) -> "TriMesh":
        """
        Factory method for periodic meshes from Gmsh.
        
        Args:
            VX, VY: Vertex coordinates
            EToV: Element-to-vertex connectivity
            BCType: Boundary condition type array from Gmsh .neu format
            order: Polynomial order
            periodic_pairs: List of (master_bc_code, slave_bc_code) pairs
            NODETOL: Tolerance for node matching
        """
        VX, VY, EToV, periodic_pairs, global_edge_to_tuple, bc_node_map = generate_periodic_square_mesh(
            h=h,
            domain_size=domain_size,
        )
        return cls.build(
            VX=VX,
            VY=VY,
            EToV=EToV,
            order=order,
            BCType=bc_node_map,
            periodic_edge_map=periodic_pairs,
            global_edge_to_tuple=global_edge_to_tuple,
            NODETOL=NODETOL
        )
    
    @classmethod
    def build(
        cls, 
        VX, VY, EToV: np.ndarray,
        BCType, 
        order: int,
        NODETOL: float = 1e-10,
        periodic_edge_map: Dict[int, int] = None,
        global_edge_to_tuple: Dict[int, tuple] = None
    ) -> "TriMesh":
        """Factory: compute geometry (one-time cost)"""
        K = EToV.shape[0]

        # Step 1: Compute nodes in reference triangle (NumPy)
        # Use recursivenodes library for nodes
        Np = (order + 1) * (order + 2) // 2
        Nfp = order + 1
        r, s, Vand, Vr, Vs = TriElements(order)
        Dr = Vr @ np.linalg.inv(Vand)
        Ds = Vs @ np.linalg.inv(Vand)
        Drw = (Vand @ Vr.T) @ np.linalg.inv( (Vand @ Vand.T) )
        Dsw = (Vand @ Vs.T) @ np.linalg.inv( (Vand @ Vand.T) )
        Dw = np.stack([Drw, Dsw], axis=-1)  # [Np, Np, 2]    
        # Step 2: Compute physical coordinates
        x, y, Fmask, Fx, Fy = _compute_physical_coordinates(EToV, VX, VY, r, s, NODETOL)

        # Step 3: Compute geometric factors
        rx, sx, ry, sy, J = _compute_geometric_factors(x, y, Dr, Ds)
        r_xy = np.stack([rx, ry], axis=-1)
        s_xy = np.stack([sx, sy], axis=-1)
        rs_xy = np.stack([r_xy, s_xy], axis=-1) # [Np, K, 2, 2]
        Lift = _Lift(order, Np, 3, Nfp, Fmask, r, s, Vand)

        # Step 4: Compute face normals and Jacobians
        face_normals, face_sJ = _compute_face_normals(Dr, Ds, x, y, Fmask, Nfp, K)
        face_scale = face_sJ / J[Fmask.flatten('F'), :]

        # Step 5: Compute connectivity (NumPy)
        EToE, EToF = _connect_elements(EToV)

        mapM, mapP, vmapM, vmapP = _build_node_maps(
            K, Np, 3, Nfp, Fmask, EToE, EToF, EToV, VX, VY, x, y, NODETOL,
            periodic_edge_map=periodic_edge_map,
            global_edge_to_tuple=global_edge_to_tuple
        )

        if BCType is not None:
            # Build bc_maps from BCType (this will exclude periodic boundaries since they're now interior)
            bc_maps = _build_bc_maps(Nfp, BCType)
        else:
            bc_maps = {}

        # Step 6: Convert to JAX arrays and freeze
        return cls(
            # Geometry
            x=jnp.array(x),
            y=jnp.array(y),
            Fx=jnp.array(Fx),
            Fy=jnp.array(Fy),
            Vand=jnp.array(Vand),
            invVand=jnp.array(np.linalg.inv(Vand)),
            Dw=jnp.array(Dw),
            rs_xy=jnp.array(rs_xy),
            face_normals=jnp.array(face_normals.reshape(Nfp, 3, K, 2, order='F')),
            face_scale=jnp.array(face_scale.reshape(Nfp, 3, K, 1, order='F')),
            Lift=jnp.array(Lift.reshape(-1, Nfp, 3, order='F')),
            J = jnp.array(J),
            face_sJ = jnp.array(face_sJ),
            # Connectivity
            EToE=EToE,
            EToF=EToF,
            mapM=mapM,
            mapP=mapP,
            vmapM=vmapM,
            vmapP=vmapP,
            bc_maps=bc_maps,
            # Metadata
            order=order,
            Np=Np,
            Nfp=Nfp,
            K=K
        )

# =====================================================================
# Helper functions
# =====================================================================

# --- Private helper functions (NumPy-only, no JAX dependency) ---
def _Lift(N, Np, Nfaces, Nfp, Fmask, r, s, V):
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
    r,s : ndarray
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
    for face in range(Nfaces-1):
        # Extract face coordinates
        faceR = r[Fmask[:, face], None]
        
        # Compute 1D Vandermonde matrix for the face
        V1D = Vandermonde(1, N, faceR)
        
        # Compute mass matrix for the edge
        massEdge = np.linalg.inv(V1D @ V1D.T)
        
        # Populate Emat for this face
        Emat[Fmask[:, face], face*Nfp:(face+1)*Nfp] = massEdge

    # Extract face coordinates
    faceS = s[Fmask[:, 2], None]
        
    # Compute 1D Vandermonde matrix for the face
    V1D = Vandermonde(1, N, faceS)
        
    # Compute mass matrix for the edge
    massEdge = np.linalg.inv(V1D @ V1D.T)
        
    # Populate Emat for this face
    Emat[Fmask[:, 2], 2*Nfp:(2+1)*Nfp] = massEdge

    # Compute LIFT matrix
    LIFT = V @ (V.T @ Emat)
        
    return LIFT


def _TriElements(order) -> Tuple[Array, Array, Array, Array]:
    """Get reference triangle nodes and differentiation matrices"""
    nodes  = warburton(2, order, domain="biunit")  # recursivenodes API
    r, s   = nodes[:, 0], nodes[:, 1]
    Vand   = Vandermonde(2, order, nodes, out=None, C=None)
    Vgrad  = VandGrad(2, order, nodes, out=None, C=None, work=None, both=False)
    Dr, Ds = Vgrad[...,0], Vgrad[...,1]  # [Np, Np]
    return r, s, Vand, Dr, Ds      


def _compute_physical_coordinates(
    EToV: np.ndarray,
    VX: np.ndarray,
    VY: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    NODETOL: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute physical coordinates of nodes (vectorized)"""
    # Extract vertex indices
    va = EToV[:, 0]
    vb = EToV[:, 1]
    vc = EToV[:, 2]
    
    # Vectorized coordinate transformation
    r = r.reshape(-1, 1)
    s = s.reshape(-1, 1)
    
    # Barycentric mapping: x = 0.5 * (-(r+s)*Va + (1+r)*Vb + (1+s)*Vc)
    x = 0.5 * (-(r + s) * VX[va] + (1 + r) * VX[vb] + (1 + s) * VX[vc])
    y = 0.5 * (-(r + s) * VY[va] + (1 + r) * VY[vb] + (1 + s) * VY[vc])
    
    # Find face masks
    fmask1 = np.where(np.abs(s + 1) < NODETOL)[0]
    fmask2 = np.where(np.abs(r + s) < NODETOL)[0]
    fmask3 = np.where(np.abs(r + 1) < NODETOL)[0]
    Fmask = np.stack([fmask1, fmask2, fmask3]).T

    Fx = x[Fmask.flatten(order='F'), :]
    Fy = y[Fmask.flatten(order='F'), :]

    return x, y, Fmask, Fx, Fy    

def _compute_geometric_factors(
    x: np.ndarray,
    y: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute metric terms and Jacobian"""
    xr = Dr @ x
    xs = Ds @ x
    yr = Dr @ y
    ys = Ds @ y
    
    J = xr * ys - xs * yr
    rx = ys / J
    sx = -yr / J
    ry = -xs / J
    sy = xr / J
    
    return rx, sx, ry, sy, J

def _compute_face_normals(
    Dr: np.ndarray,
    Ds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    Fmask: np.ndarray,
    Nfp: int,
    K: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute outward pointing normals at elements faces and surface Jacobians
    Parameters:
    Dr : numpy array
        Derivative matrix in r direction
    Ds : numpy array
        Derivative matrix in s direction
    x : numpy array
        x coordinates
    y : numpy array
        y coordinates
    Fmask : numpy array
        Face mask indices
    Nfp : int
        Number of face points
    K : int
        Number of elements
    Returns:
    nx, ny : numpy arrays
        Normalized normal vectors
    sJ : numpy array
        Surface Jacobians
    """
    # Compute geometric factors
    xr = Dr @ x
    yr = Dr @ y
    xs = Ds @ x
    ys = Ds @ y
    J = xr * ys - xs * yr
    Fmask = Fmask.flatten(order='F')
    # Interpolate geometric factors to face nodes
    fxr = xr[Fmask, :]
    fxs = xs[Fmask, :]
    fyr = yr[Fmask, :]
    fys = ys[Fmask, :]
    # Initialize normal vectors
    nx = np.zeros((3*Nfp, K))
    ny = np.zeros((3*Nfp, K))
    # Define face indices
    fid1 = np.arange(Nfp)
    fid2 = np.arange(Nfp, 2*Nfp)
    fid3 = np.arange(2*Nfp, 3*Nfp)
    # Face 1
    nx[fid1, :] = fyr[fid1, :]
    ny[fid1, :] = -fxr[fid1, :]
    # Face 2
    nx[fid2, :] = fys[fid2, :] - fyr[fid2, :]
    ny[fid2, :] = -fxs[fid2, :] + fxr[fid2, :]
    # Face 3
    nx[fid3, :] = -fys[fid3, :]
    ny[fid3, :] = fxs[fid3, :]
    # Normalize
    sJ = np.sqrt(nx**2 + ny**2)
    nx = nx / sJ
    ny = ny / sJ
    
    normals = np.stack([nx,
                        ny], axis=-1)
    
    return normals, sJ

def _connect_elements(EToV: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangle face connect algorithm due to Toby Isaac
    Parameters:
    EToV : numpy array
        Element to vertex connectivity (0-indexed)
    Returns:
    EToE : numpy array
        Element to element connectivity
    EToF : numpy array
        Element to face connectivity
    """
    Nfaces = 3
    K = EToV.shape[0]
    Nnodes = np.max(EToV) + 1  # +1 because we're 0-indexed

    # create list of all faces 1, then 2, & 3
    fnodes = np.vstack([
        EToV[:, [0, 1]],
        EToV[:, [1, 2]],
        EToV[:, [2, 0]]
    ])

    # Sort the nodes but don't subtract 1 since we're already 0-indexed
    fnodes = np.sort(fnodes, axis=1, kind='stable')

    # set up default element to element and Element to faces connectivity
    # Use 0-indexed arrays for Python
    EToE = np.tile(np.arange(K), (Nfaces, 1)).T
    EToF = np.tile(np.arange(Nfaces), (K, 1))

    # uniquely number each set of three faces by their node numbers
    # Adjust the formula for 0-indexed nodes
    id = fnodes[:, 0] * Nnodes + fnodes[:, 1]

    # Create spNodeToNode with 0-indexed values
    spNodeToNode = np.column_stack([
        id, 
        np.arange(Nfaces*K),  # 0-indexed
        EToE.flatten(order='F'),
        EToF.flatten(order='F')
    ])

    # Now we sort by global face number
    sorted_indices = np.argsort(spNodeToNode[:, 0], kind='stable')
    sorted_spNodeToNode = spNodeToNode[sorted_indices]

    # find matches in the sorted face list
    indices = np.where(sorted_spNodeToNode[:-1, 0] == sorted_spNodeToNode[1:, 0])[0]

    # make links reflexive
    matchL = np.vstack([
        sorted_spNodeToNode[indices],
        sorted_spNodeToNode[indices+1]
    ])
    matchR = np.vstack([
        sorted_spNodeToNode[indices+1],
        sorted_spNodeToNode[indices]
    ])

    # Insert matches using linear indexing
    linear_indices = matchL[:, 1].astype(int)

    # Flatten arrays in column-major order for direct indexing
    EToE_flat = EToE.flatten(order='F')
    EToF_flat = EToF.flatten(order='F')

    # Update using linear indexing
    EToE_flat[linear_indices] = matchR[:, 2]
    EToF_flat[linear_indices] = matchR[:, 3]

    # Reshape back to original form
    EToE = EToE_flat.reshape(EToE.shape, order='F')
    EToF = EToF_flat.reshape(EToF.shape, order='F')

    return EToE, EToF

def _build_node_maps(
    K: int, Np: int, Nfaces: int, Nfp: int, Fmask: np.ndarray,
    EToE: np.ndarray, EToF: np.ndarray, EToV: np.ndarray,
    VX: np.ndarray, VY: np.ndarray, x: np.ndarray, y: np.ndarray,
    NODETOL: float,
    periodic_edge_map: Dict[int, int] = None,
    global_edge_to_tuple: Dict[int, tuple] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ================================================================
    # Vectorized initialization
    # ================================================================
    nodeids = np.arange(0, K * Np).reshape(Np, K, order='F')
    
    # Precompute vmapM vectorized
    vmapM = np.zeros((Nfp, Nfaces, K), dtype=int)
    for f1 in range(Nfaces):
        vmapM[:, f1, :] = nodeids[Fmask[:, f1], :]
    
    # Initialize vmapP and mapP
    vmapP = np.zeros((Nfp, Nfaces, K), dtype=int)
    mapP = np.arange(0, K * Nfp * Nfaces).reshape((Nfp, Nfaces, K), order='F')
    
    # ================================================================
    # Step 1: Vectorized distance matching for non-periodic boundaries
    # ================================================================
    # Pre-flatten x and y for faster indexing
    x_flat = x.ravel(order='F')
    y_flat = y.ravel(order='F')
    
    # Get all face pairs at once
    # Create arrays of all (k1, f1) pairs
    k1_all, f1_all = np.meshgrid(np.arange(K), np.arange(Nfaces), indexing='ij')
    k1_all = k1_all.ravel()
    f1_all = f1_all.ravel()
    
    # Get corresponding neighbors
    k2_all = EToE[k1_all, f1_all]
    f2_all = EToF[k1_all, f1_all]
    
    # Precompute reference edge lengths for all faces
    # Vectorized computation of reference lengths
    v1_all = EToV[k1_all, f1_all]
    v2_all = EToV[k1_all, 1 + f1_all % (Nfaces - 1)]
    refd_all = np.sqrt((VX[v1_all] - VX[v2_all])**2 + (VY[v1_all] - VY[v2_all])**2)
    
    # Process in batches to avoid excessive memory usage
    batch_size = min(1000, len(k1_all))  # Adjust based on available memory
    n_batches = (len(k1_all) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(k1_all))
        
        batch_k1 = k1_all[start_idx:end_idx]
        batch_f1 = f1_all[start_idx:end_idx]
        batch_k2 = k2_all[start_idx:end_idx]
        batch_f2 = f2_all[start_idx:end_idx]
        batch_refd = refd_all[start_idx:end_idx]
        
        # Get node indices for master and neighbor faces
        vidM_batch = vmapM[:, batch_f1, batch_k1]  # Shape: (Nfp, batch_size)
        vidP_batch = vmapM[:, batch_f2, batch_k2]  # Shape: (Nfp, batch_size)
        
        # Get coordinates
        x1 = x_flat[vidM_batch]  # Shape: (Nfp, batch_size)
        y1 = y_flat[vidM_batch]
        x2 = x_flat[vidP_batch]  # Shape: (Nfp, batch_size)
        y2 = y_flat[vidP_batch]
        
        # Vectorized distance computation using broadcasting
        # Instead of computing full Nfp x Nfp matrix, we can compute pairwise distances
        # with reduced memory footprint
        
        # For each point in vidM, find the closest point in vidP
        for i in range(Nfp):
            # Compute distances from i-th point in master to all points in neighbor
            dx = x1[i, :, np.newaxis] - x2.T  # Shape: (batch_size, Nfp)
            dy = y1[i, :, np.newaxis] - y2.T
            D_sq = dx**2 + dy**2
            
            # Find minimum distance for each face in batch
            min_dist_idx = np.argmin(D_sq, axis=1)
            min_dist = np.min(D_sq, axis=1)
            
            # Check if within tolerance
            mask = np.sqrt(min_dist) < NODETOL * batch_refd
            
            # Update vmapP and mapP for matching pairs
            if np.any(mask):
                vmapP[i, batch_f1[mask], batch_k1[mask]] = vidP_batch[min_dist_idx[mask], np.arange(np.sum(mask))]
                mapP[i, batch_f1[mask], batch_k1[mask]] = (min_dist_idx[mask] + 
                                                          batch_f2[mask] * Nfp + 
                                                          batch_k2[mask] * Nfaces * Nfp)
    
    # ================================================================
    # Step 2: Optimized periodic boundary connections
    # ================================================================
    if periodic_edge_map is not None and global_edge_to_tuple is not None:
        # Precompute edge_to_face_map for boundary faces only
        # Find boundary faces where EToE[k, f] == k
        boundary_mask = (EToE == np.arange(K)[:, np.newaxis])
        boundary_k, boundary_f = np.where(boundary_mask)
        
        # Build edge_to_face_map only for boundary faces
        edge_to_face_map = {}
        for k, f in zip(boundary_k, boundary_f):
            v1 = EToV[k, f]
            v2 = EToV[k, (f + 1) % 3]
            edge_tuple = tuple(sorted([v1, v2]))
            
            if edge_tuple not in edge_to_face_map:
                edge_to_face_map[edge_tuple] = []
            edge_to_face_map[edge_tuple].append((k, f))
        
        # Process periodic connections
        processed_edges = set()
        periodic_pairs = list(periodic_edge_map.items()) if hasattr(periodic_edge_map, 'items') else periodic_edge_map
        
        for edge1_global, edge2_global in periodic_pairs:
            if edge1_global in processed_edges:
                continue
                
            edge1_tuple = global_edge_to_tuple.get(edge1_global)
            edge2_tuple = global_edge_to_tuple.get(edge2_global)
            
            if edge1_tuple is None or edge2_tuple is None:
                continue
                
            faces1 = edge_to_face_map.get(edge1_tuple, [])
            faces2 = edge_to_face_map.get(edge2_tuple, [])
            
            if len(faces1) == 1 and len(faces2) == 1:
                k1, f1 = faces1[0]
                k2, f2 = faces2[0]
                
                vidM1 = vmapM[:, f1, k1]
                vidM2 = vmapM[:, f2, k2]
                
                x1_coords = x_flat[vidM1]
                y1_coords = y_flat[vidM1]
                x2_coords = x_flat[vidM2]
                y2_coords = y_flat[vidM2]
                
                # Determine periodicity direction
                mean_y1, mean_y2 = np.mean(y1_coords), np.mean(y2_coords)
                mean_x1, mean_x2 = np.mean(x1_coords), np.mean(x2_coords)
                
                if np.abs(mean_y1 - mean_y2) < np.sqrt(NODETOL):
                    # Horizontal periodicity
                    idx1 = np.argsort(y1_coords)
                    idx2 = np.argsort(y2_coords)
                elif np.abs(mean_x1 - mean_x2) < np.sqrt(NODETOL):
                    # Vertical periodicity
                    idx1 = np.argsort(x1_coords)
                    idx2 = np.argsort(x2_coords)
                else:
                    # Fallback: assume natural ordering
                    idx1 = np.arange(Nfp)
                    idx2 = np.arange(Nfp)
                
                # Apply mapping
                vmapP[idx1, f1, k1] = vidM2[idx2]
                vmapP[idx2, f2, k2] = vidM1[idx1]
                
                mapP[idx1, f1, k1] = idx2 + f2 * Nfp + k2 * Nfaces * Nfp
                mapP[idx2, f2, k2] = idx1 + f1 * Nfp + k1 * Nfaces * Nfp
                
                processed_edges.add(edge1_global)
                processed_edges.add(edge2_global)
    
    # ================================================================
    # Final reshaping
    # ================================================================
    vmapP_flat = vmapP.ravel(order="F")
    vmapM_flat = vmapM.ravel(order="F")
    mapP_flat = mapP.ravel(order="F")
    
    # For surface flux computation
    mapM = vmapM_flat.reshape(Nfp * Nfaces, K, order="F")
    mapP = vmapP_flat.reshape(Nfp * Nfaces, K, order="F")
    
    return mapM, mapP, vmapM_flat, vmapP_flat

def _build_bc_maps(
    Nfp: int, BCType: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build boundary condition node maps"""
    # BCType: [Nfaces, K]
    bc_maps = {}
    
    # Flatten for easier indexing
    bct = BCType.T  # [K, Nfaces]
    bnodes = np.outer(np.ones(Nfp), bct.flatten(order='F')).astype(int)
    bnodes = bnodes.flatten(order='F')
    
    # Define BC names including periodic boundaries
    BC_NAMES = {
        1: "in", 2: "out", 3: "wall", 4: "far",
        5: "cylinder", 6: "dirichlet", 7: "neuman", 8: "slip",
    }
    
    for code, name in BC_NAMES.items():
        mask = np.where(bnodes == code)[0]
        if len(mask) > 0:
            bc_maps[name] = mask
    
    return bc_maps    

def _get_bc_name(code: int) -> str:
    """Get boundary condition name from code"""
    BC_NAMES = {
        1: "in", 2: "out", 3: "wall", 4: "far",
        5: "cylinder", 6: "dirichlet", 7: "neuman", 8: "slip",
    }
    return BC_NAMES.get(code, f"bc_{code}")