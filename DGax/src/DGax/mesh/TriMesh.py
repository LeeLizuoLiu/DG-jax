# src/DGax/mesh/TriMesh.py
"""Triangular mesh Management"""
from dataclasses import dataclass
from typing import Tuple, Dict
import jax.numpy as jnp
from jax import Array
import numpy as np
from recursivenodes.nodes import warburton
from recursivenodes.polynomials import proriolkoornwinderdubinervandermondegrad as VandGrad
from recursivenodes.polynomials import proriolkoornwinderdubinervandermonde as Vandermonde
from .mesh import Mesh
from .io import read_gambit_neu

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
    # J: Array                   # [Np, K] - Volume Jacobian
    Dw: Array                    # Weak differentiation matrix [dr, ds]
    rs_xy: Array                 # [dr/dx, dr/dy]
                                 # [ds/dx, ds/dy]
    face_normals: Array          # [Nfp*Nfaces, K, 2] - Face normals (nx, ny)
    # face_sJ: Array             # [Nfp, Nfaces, K] - Surface Jacobian
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
    
    def get_boundary_mask(self, bc_type: str) -> Array:
        """Get linear indices for boundary type"""
        return self.bc_maps.get(bc_type, jnp.array([], dtype=int))
    
    @classmethod
    def from_gambit(
        cls, 
        mesh_path: str, 
        order: int,
        NODETOL: float = 1e-12
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
    def build(
        cls, 
        VX, VY, EToV: np.ndarray,
        BCType, 
        order: int,
        NODETOL: float = 1e-12
    ) -> "TriMesh":
        """Factory: compute geometry (one-time cost)"""
        K = EToV.shape[0]
        
        # Step 1: Compute nodes in reference triangle (NumPy)
        # Use recursivenodes library for nodes
        Np = (order + 1) * (order + 2) // 2
        Nfp = order + 1
        r, s, Vand, Vr, Vs = _TriElements(order)
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
            K, Np, 3, Nfp, Fmask, EToE, EToF, EToV, VX, VY, x, y, NODETOL
        )
        bc_maps = _build_bc_maps(Nfp, BCType, vmapM)
        
        # Step 6: Convert to JAX arrays and freeze
        return cls(
            # Geometry
            x=jnp.array(x),
            y=jnp.array(y),
            Fx=jnp.array(Fx),
            Fy=jnp.array(Fy),
            Vand=jnp.array(Vand),
            invVand = jnp.array(np.linalg.inv(Vand)),
            # J=jnp.array(J),
            Dw=jnp.array(Dw),
            rs_xy=jnp.array(rs_xy),
            face_normals=jnp.array(face_normals.reshape(Nfp, 3, K, 2, order='F')),
            # face_sJ=jnp.array(face_sJ),
            face_scale=jnp.array(face_scale.reshape(Nfp, 3, K, 1, order='F')),
            Lift=jnp.array(Lift.reshape(-1, Nfp, 3, order='F')),
            # Connectivity
            # EToV=jnp.array(EToV),
            EToE=EToE,
            EToF=EToF,
            mapM=mapM,
            mapP=mapP,
            vmapM=vmapM,
            vmapP=vmapP,
            bc_maps= bc_maps,
            # Metadata
            order=order,
            Np=Np,
            Nfp=Nfp,
            K=K
        )

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
    NODETOL: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # number volume nodes consecutively
    nodeids = np.arange(0, K*Np).reshape(Np, K, order='F')

    vmapM = np.zeros((Nfp, Nfaces, K), dtype=int)
    vmapP = np.zeros((Nfp, Nfaces, K), dtype=int)

    mapM = np.arange(0, K*Nfp*Nfaces)
    mapP = np.arange(0, K*Nfp*Nfaces).reshape((Nfp, Nfaces, K), order='F')
    # find index of face nodes with respect to volume node ordering
    for k1 in range(K):
        for f1 in range(Nfaces):
            vmapM[:, f1, k1] = nodeids[Fmask[:, f1], k1]

    for k1 in range(K):
        for f1 in range(Nfaces):
            # find neighbor
            """
            In short, **`f2` is the local index of the face in the neighboring element (`k2`) that is shared with the current element (`k1`)**.

            Think of it like two rooms sharing a wall.
            
              * `k1` is your current room.
              * `f1` is one of the walls in your room (e.g., your "north wall").
              * `k2` is the room on the other side of that wall.
              * `f2` is what the *neighbor* calls that same shared wall (e.g., their "south wall").
            
            The `EToF` (Element-to-Face) array is like a map that tells you this information. `EToF[k1, f1]` looks up which of the neighbor's walls corresponds to wall `f1` of your current room `k1`.
            """
            k2 = EToE[k1, f1]
            f2 = EToF[k1, f1]

            # reference length of edge
            v1 = EToV[k1, f1]
            v2 = EToV[k1, 1 + f1 % (Nfaces - 1)]

            refd = np.sqrt((VX[v1] - VX[v2])**2 + (VY[v1] - VY[v2])**2)

            # find volume node numbers of left and right nodes
            vidM = vmapM[:, f1, k1]
            vidP = vmapM[:, f2, k2]
            x1 = x.ravel(order='F')[vidM]
            y1 = y.ravel(order='F')[vidM]
            x2 = x.ravel(order='F')[vidP]
            y2 = y.ravel(order='F')[vidP]

            # Compute distance matrix
            D = (x1[:, np.newaxis] - x2)**2 + (y1[:, np.newaxis] - y2)**2
            # Find indices where distance is small
            idP, idM = np.where(np.sqrt(np.abs(D)) < NODETOL * refd)

            vmapP[idM, f1, k1] = vidP[idP]
            mapP[idM, f1, k1] = idP + (f2)*Nfp + (k2)*Nfaces*Nfp

    # reshape vmapM and vmapP to be vectors and create boundary node list
    vmapP = vmapP.ravel(order="F")
    vmapM = vmapM.ravel(order="F")
    mapP =   mapP.ravel(order="F")

    # for surface flux computation
    mapM = vmapM.reshape(Nfp*Nfaces, K, order="F")
    mapP = vmapP.reshape(Nfp*Nfaces, K, order="F")
    return mapM, mapP, vmapM, vmapP

def _build_bc_maps(
    Nfp: int, BCType: np.ndarray, vmapM: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build boundary condition node maps"""
    # BCType: [Nfaces, K]
    bc_maps = {}
    
    # Flatten for easier indexing
    bct = BCType.T  # [K, Nfaces]
    bnodes = np.outer(np.ones(Nfp), bct.flatten(order='F')).astype(int)
    bnodes = bnodes.flatten(order='F')
    # Define BC names
    BC_NAMES = {
        1: "in", 2: "out", 3: "wall", 4: "far",
        5: "cylinder", 6: "dirichlet", 7: "neuman", 8: "slip"
    }
    
    for code, name in BC_NAMES.items():
        mask = np.where(bnodes == code)[0]
        bc_maps[name] = mask
    
    return bc_maps        