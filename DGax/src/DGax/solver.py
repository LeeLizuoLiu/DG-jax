import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap
from typing import Callable, Dict
from .boundary_conditions.boundary_conditions import BoundaryCondition
from functools import partial

class DGSolver(eqx.Module):
    equations: eqx.Module
    riemann_solver: Callable
    mesh: eqx.Module  # TriMesh object
    boundary_conditions: Dict[str, BoundaryCondition]  # bc_name -> BC object
    
    cfl: float = 0.5
    
    def rhs(self, u: Array, t: float) -> Array:
        """Compute du/dt with BC enforcement"""
        # Volume integral (unchanged)
        du_vol = self._volume_integral(u)
        
        # Surface integral with BCs
        du_surf = self._surface_integral_with_bc(u, t)
        
        return du_vol + du_surf
    
    def _surface_integral_with_bc(self, u: Array, t: float) -> Array:
        """Surface flux computation with boundary treatment"""
        # Extract interior face values: [Nfp, Nfaces, K]
        u_left = self._extract_faces(u)
        
        # Get neighbor states (before BC modification)
        u_right = self._get_neighbor_states(u_left)
        
        # Apply boundary conditions where needed
        u_right = self._apply_boundary_conditions(u_left, u_right, t)
        
        # Compute flux with modified exterior state
        flux_faces = self._compute_numerical_flux(u_left, u_right)
        
        # Lift to volume
        return self.mesh.Lift @ flux_faces.reshape(-1, self.mesh.n_elements, order='F')
    
    def _get_neighbor_states(self, u_left: Array) -> Array:
        """Get neighbor state (including self for boundaries)"""
        u_flat = u_left.reshape(-1, order='F')
        return u_flat[self.mesh.vmapP].reshape(
            self.mesh.Nfp, self.mesh.Nfaces, self.mesh.n_elements, order='F'
        )
    
    def _apply_boundary_conditions(
        self,
        u_left: Array,
        u_right: Array,
        t: float
    ) -> Array:
        """
        Transform u_right for boundary faces using BC objects.
        Uses vectorized operations over all BC types.
        """
        # Initialize modified exterior state
        u_right_bc = u_right
        
        # Iterate over boundary types (e.g., "inlet", "wall")
        for bc_type, bc_obj in self.boundary_conditions.items():
            # Get linear indices for this BC type
            face_indices = self.mesh.get_boundary_mask(bc_type)
            
            if len(face_indices) == 0:
                continue
            
            # Reshape indices for 3D array indexing
            # face_indices are linear in [Nfp, Nfaces, K] ordering
            idx_f, idx_face, idx_elem = jnp.unravel_index(
                face_indices, (self.mesh.Nfp, self.mesh.Nfaces, self.mesh.n_elements)
            )
            
            # Extract interior states for these boundary faces
            u_interior = u_left[idx_f, idx_face, idx_elem]
            
            # Get corresponding normals
            normals = self.mesh.face_normals[idx_f, idx_face, idx_elem]
            
            # Apply BC transformation (vectorized over selected faces)
            u_exterior = vmap(
                lambda ui, n: bc_obj(ui, n, self.equations, t)
            )(u_interior, normals)
            
            # Update u_right at boundary faces
            u_right_bc = u_right_bc.at[idx_f, idx_face, idx_elem].set(u_exterior)
        
        return u_right_bc
    
    def _compute_numerical_flux(self, u_left: Array, u_right: Array) -> Array:
        """Vectorized Riemann solver"""
        solver = partial(self.riemann_solver, equations=self.equations)
        
        return vmap(
            vmap(
                vmap(solver, in_axes=(0, 0, 0)),
                in_axes=(0, 0, 0)
            ),
            in_axes=(2, 2, 2)
        )(u_left, u_right, self.mesh.face_normals)
    
    def _extract_faces(self, u: Array) -> Array:
        u_flat = u.reshape(-1, order='F')
        return u_flat[self.mesh.vmapM].reshape(
            self.mesh.Nfp, self.mesh.Nfaces, self.mesh.n_elements, order='F'
        )
    
    def _volume_integral(self, u: Array) -> Array:
        """Weak form volume integral
        u.shape = [Np, K, n_vars]
        flux.shape = [Np, K, n_vars, 2]
        Dw = [Drw, Dsw] shape = [Np, Np, 2]
        weak_flux = jnp.einsum("jknm,pjd->pknmd", flux, Dw) [Np, K, n_vars, 2, 2]
                                                                            [dFdr  dFds]
                                                                            [dGdr  dGds]
        Jrs_xy = [[rx sx]
                  [ry sy]]  shape = [Np, K, 2, 2]
        volume_flux = jnp.einsum("pknmd, pkmd->pkn", weak_flux, Jrs_xy)
        """
        flux = self.equations.flux(u)
        dFdr =  self.mesh.Drw @ flux[...,0]
        dFds =  self.mesh.Dsw @ flux[...,0]
        dGdr =  self.mesh.Drw @ flux[...,1]
        dGds =  self.mesh.Dsw @ flux[...,1]
        dFdx = self.mesh.rx * dFdr + self.mesh.sx * dFds
        dGdy = self.mesh.ry * dGdr + self.mesh.sy * dGds
        return   dFdx + dGdy