import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap
from typing import Callable, Dict
from .boundary_conditions.boundary_conditions import BoundaryCondition
from .mesh.mesh import Mesh
from functools import partial

class DGSolver(eqx.Module):
    equations: eqx.Module
    riemann_solver: Callable
    mesh: Mesh  # TriMesh object
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
        u_flat = u.reshape(-1, 4, order='F')
        # Extract interior face values
        uM = self._extract_faces(u_flat)
        
        # Get neighbor states (before BC modification)
        uP = self._get_neighbor_states(u_flat)
        
        # Apply boundary conditions where needed
        uP = self._apply_boundary_conditions(uM, uP, t)
        
        # Compute flux with modified exterior state
        flux_faces = self._compute_numerical_flux(uM, uP)
        
        return - jnp.einsum('inf,nfkl->ikl',self.mesh.Lift, self.mesh.face_scale/2 * flux_faces)

    def _get_neighbor_states(self, u_flat: Array) -> Array:
        """Get neighbor state (including self for boundaries)"""
        return u_flat[self.mesh.mapP,:]
    
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
    
    def _compute_numerical_flux(self, uM: Array, uP: Array) -> Array:
        """Vectorized Riemann solver"""
        solver = partial(self.riemann_solver, equations=self.equations)
        
        face_flux = vmap(
                              vmap(solver ,in_axes=(1,1,1,None), out_axes=1),
                            in_axes=(1,1,1,None), out_axes=1)(uM, uP, self.mesh.face_normals)
        return face_flux


    
    def _extract_faces(self, u_flat: Array) -> Array:
        return u_flat[self.mesh.mapM, :]
    
    def _volume_integral(self, u: Array) -> Array:
        """Weak form volume integral
        u.shape = [Np, K, n_vars]
        flux.shape = [Np, K, n_vars, 2]
        Dw = [Drw, Dsw] shape = [Np, Np, 2]
        weak_flux = jnp.einsum("jknm,pjd->pknmd", flux, Dw) [Np, K, n_vars, 2, 2]
                                                                            [dFdr  dFds]
                                                                            [dGdr  dGds]
        Jrs_xy = [[rx ry]
                  [sx sy]]  shape = [Np, K, 2, 2]
        volume_flux = jnp.einsum("pknmd, pkmd->pkn", weak_flux, Jrs_xy)
        """
        flux = self.equations.flux(u)
        weak_flux = jnp.einsum("jknm,pjd->pknmd", flux, self.mesh.Dw)
        volume_integral = jnp.einsum("pknmd, pkmd->pkn", weak_flux, self.mesh.rs_xy)
        return  volume_integral 