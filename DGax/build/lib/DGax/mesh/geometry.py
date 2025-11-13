# src/dg_jax/mesh/geometry.py
"""Mesh geometry calculations (coordinates, Jacobians, normals)"""
from dataclasses import dataclass
from typing import Tuple
import jax.numpy as jnp
from jax import Array
import numpy as np
import math
from recursivenodes.nodes import warburton

@dataclass(frozen=True)
class MeshGeometry2D:
    """Immutable container for 2D mesh geometry"""
    x: Array                 # [Np, K] - physical coordinates 
    y: Array                 # [Np, K]
    J: Array                 # [Np, K] - Jacobian
    rx: Array                # metric term: dr/dx
    sx: Array                # ds/dx
    ry: Array                # dr/dy
    sy: Array                # ds/dy
    face_normals: Array      # [Nfp, Nfaces, K, 2] - face normals
    face_sJ: Array          # [Nfp, Nfaces, K] - face Jacobian
    face_scale: Array       # [Nfp, Nfaces, K] - Fscale
    
    @classmethod
    def from_gambit_and_order(
        cls, 
        mesh_path: str, 
        order: int,
        NODETOL: float = 1e-12
    ) -> "MeshGeometry2D":
        """Factory: read file + compute geometry (one-time cost)"""
        # Step 1: Read raw mesh data (NumPy)
        from .io import read_gambit_neu
        VX, VY, EToV, BCType = read_gambit_neu(mesh_path)
        K = EToV.shape[0]
        
        # Step 2: Compute nodes in reference triangle (NumPy)
        # Use recursivenodes library for nodes
        Np = (order + 1) * (order + 2) // 2
        nodes = warburton(2, order, domain="unit")  # recursivenodes API
        r, s = nodes[:, 0], nodes[:, 1]
        # Step 3: Compute physical coordinates
        x, y, Fmask = compute_physical_coordinates(EToV, VX, VY, r, s, NODETOL)
        
        # Step 4: Compute geometric factors
        # Build differentiation matrices using recursivenodes
        from recursivenodes import differentiation_matrices
        Dr, Ds = differentiation_matrices(order, domain="unit")
        rx, sx, ry, sy, J = compute_geometric_factors(x, y, Dr, Ds)
        
        # Step 5: Compute face normals and Jacobians
        face_normals, face_sJ = compute_face_normals(Dr, Ds, x, y, Fmask, order, K)
        Fscale = face_sJ / J[Fmask.flatten('F'), :]
        
        # Step 6: Convert to JAX arrays and freeze
        return cls(
            x=jnp.array(x),
            y=jnp.array(y),
            J=jnp.array(J),
            rx=jnp.array(rx),
            sx=jnp.array(sx),
            ry=jnp.array(ry),
            sy=jnp.array(sy),
            face_normals=jnp.array(face_normals),
            face_sJ=jnp.array(face_sJ),
            face_scale=jnp.array(Fscale)
        )

def compute_physical_coordinates(
    EToV: np.ndarray,
    VX: np.ndarray,
    VY: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    NODETOL: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute physical coordinates of nodes (vectorized)"""
    K = EToV.shape[0]
    Np = len(r)
    
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
    
    return x, y, Fmask

def compute_geometric_factors(
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

def compute_face_normals(
    Dr: np.ndarray,
    Ds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    Fmask: np.ndarray,
    order: int,
    K: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute face normals and surface Jacobians"""
    Nfp = order + 1
    Nfaces = 3
    
    # Interpolate geometric factors to faces
    Fmask_flat = Fmask.flatten('F')
    xr = Dr @ x
    xs = Ds @ x
    yr = Dr @ y
    ys = Ds @ y
    
    fxr = xr[Fmask_flat, :]
    fxs = xs[Fmask_flat, :]
    fyr = yr[Fmask_flat, :]
    fys = ys[Fmask_flat, :]
    
    # Initialize normals
    nx = np.zeros((Nfp * Nfaces, K))
    ny = np.zeros((Nfp * Nfaces, K))
    
    # Face 1: s = -1
    nx[:Nfp, :] = fyr[:Nfp, :]
    ny[:Nfp, :] = -fxr[:Nfp, :]
    
    # Face 2: r + s = 0
    nx[Nfp:2*Nfp, :] = fys[Nfp:2*Nfp, :] - fyr[Nfp:2*Nfp, :]
    ny[Nfp:2*Nfp, :] = -fxs[Nfp:2*Nfp, :] + fxr[Nfp:2*Nfp, :]
    
    # Face 3: r = -1
    nx[2*Nfp:, :] = -fys[2*Nfp:, :]
    ny[2*Nfp:, :] = fxs[2*Nfp:, :]
    
    # Normalize
    sJ = np.sqrt(nx**2 + ny**2)
    nx = nx / sJ
    ny = ny / sJ
    
    # Reshape to [Nfp, Nfaces, K, 2]
    normals = np.stack([nx.reshape(Nfp, Nfaces, K, order='F'),
                        ny.reshape(Nfp, Nfaces, K, order='F')], axis=-1)
    sJ = sJ.reshape(Nfp, Nfaces, K, order='F')
    
    return normals, sJ