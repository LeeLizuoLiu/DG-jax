import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional

import abc
import basix
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
import pdb
from Mesh import DGMesh, TriangularMesh, TetrahedralMesh

class Element(abc.ABC):
    """
    Abstract base class for Discontinuous Galerkin finite elements.
    Provides core functionality for DG methods with basix integration.
    Abstract methods like volume quadrature rules and facet quadrature rules should be put a place holder here
    In the detailed implementation of specific elements (e.g., triangles, tetrahedra), these methods must be implemented.
    """
    
    def __init__(self, mesh: DGMesh, degree: int, quadrature_degree: Optional[int] = None):
        """
        Initialize DG element.
        
        Args:
            mesh: DGMesh instance providing geometry and connectivity
            degree: Polynomial degree of the element
            quadrature_degree: Quadrature degree (default: 2*degree)
        """
        self.mesh = mesh
        self.degree = degree
        self.quadrature_degree = quadrature_degree if quadrature_degree is not None else 2 * degree
        
        # Basix element and quadrature
        self.element = None
        self.quadrature_points = None  # Quadrature points on reference element
        self.quadrature_weights = None  # Quadrature weights
        
        # Tabulated basis functions
        self.basis_values = None  # Shape: (num_quad, num_basis)
        self.basis_gradients_ref = None  # Shape: (num_quad, num_basis, dim)
        
        # Element properties
        self.num_basis_functions = None
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize element, quadrature, and tabulate basis."""
        self.create_element()
        self.create_quadrature()
        self.tabulate_basis()
        self.num_basis_functions = self.element.dim
    
    @abc.abstractmethod
    def create_element(self):
        """Create basix element based on cell type and degree."""
        pass
    
    @abc.abstractmethod
    def create_quadrature(self):
        """Create quadrature rule for the reference element."""
        pass
    
    def tabulate_basis(self):
        """Tabulate basis functions and gradients at quadrature points."""
        points = self.quadrature_points
        
        # Tabulate with 1 derivative (0th order is values, 1st order is gradients)
        # The output shape is (deriv_order+1, num_points, num_basis_functions, value_size)
        # For scalar elements, value_size = 1
        tab = self.element.tabulate(1, points)
        
        # Extract values: shape (num_points, num_basis)
        # tab[0] is values, shape: (num_points, num_basis, value_size)
        self.basis_values = tab[0, :, :, 0]  # Remove value_size dimension
        
        # Extract gradients: shape (num_points, num_basis, dim)
        # tab[1:dim+1] are derivatives in each direction
        # Original shape: (dim, num_points, num_basis, value_size)
        # We want: (num_points, num_basis, dim)
        self.basis_gradients_ref = tab[1:1+self.mesh.dim, :, :, 0].transpose(1, 2, 0)
    
    def compute_element_jacobians(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian matrices and determinants for all elements.
        
        Returns:
            Tuple of (jacobian_matrices, determinants)
            jacobian_matrices: shape (num_cells, dim, dim)
            determinants: shape (num_cells,)
        """
        num_cells = len(self.mesh.cells)
        dim = self.mesh.dim
        
        J_all = np.zeros((num_cells, dim, dim))
        detJ_all = np.zeros(num_cells)
        
        for cell_idx in range(num_cells):
            vertex_indices = self.mesh.cells[cell_idx]
            physical_vertices = self.mesh.vertices[vertex_indices]
            
            if dim == 2:
                # Triangle: vertices are (v0, v1, v2)
                # Jacobian columns are v1-v0 and v2-v0
                J_all[cell_idx, :, 0] = physical_vertices[1] - physical_vertices[0]
                J_all[cell_idx, :, 1] = physical_vertices[2] - physical_vertices[0]
                detJ_all[cell_idx] = np.abs(np.linalg.det(J_all[cell_idx]))
            elif dim == 3:
                # Tetrahedron: vertices are (v0, v1, v2, v3)
                # Jacobian columns are v1-v0, v2-v0, v3-v0
                J_all[cell_idx, :, 0] = physical_vertices[1] - physical_vertices[0]
                J_all[cell_idx, :, 1] = physical_vertices[2] - physical_vertices[0]
                J_all[cell_idx, :, 2] = physical_vertices[3] - physical_vertices[0]
                detJ_all[cell_idx] = np.abs(np.linalg.det(J_all[cell_idx]))
        
        return J_all, detJ_all
    
    def get_mass_matrices(self) -> jnp.ndarray:
        """
        Compute local mass matrices for all elements.
        
        Returns:
            Array of shape (num_cells, num_basis_functions, num_basis_functions)
        """
        # Compute reference mass matrix: M_ref[i,j] = ∫ φ_i φ_j dΩ_ref
        M_ref = np.einsum('q,qi,qj->ij', 
                         self.quadrature_weights,
                         self.basis_values,
                         self.basis_values)
        
        # Compute Jacobians
        _, detJs = self.compute_element_jacobians()
        
        # Scale by determinant for each element: M_e = M_ref * |detJ_e|
        num_cells = len(self.mesh.cells)
        M_all = M_ref[None, :, :] * detJs[:, None, None]
        
        return jnp.array(M_all)
    
    def get_stiffness_matrices(self) -> jnp.ndarray:
        """
        Compute local stiffness matrices for all elements.

        Returns:
            Array of shape (num_cells, num_basis_functions, num_basis_functions)
        """
        # Compute Jacobians and inverse Jacobians
        J_all, detJ_all = self.compute_element_jacobians()
        invJ_all = np.linalg.inv(J_all)

        # Transform gradients to physical space: ∇_x = J^{-T} ∇_ξ
        # basis_gradients_ref: (num_quad, num_basis, dim)
        # invJ_all: (num_cells, dim, dim)
        # We need J^{-T}: transpose last two dimensions
        invJ_T = invJ_all.transpose(0, 2, 1)  # Shape: (num_cells, dim, dim)

        # Compute physical gradients: (num_cells, num_quad, num_basis, dim)
        grad_phy = np.einsum('cij,qbj->cqbi', invJ_T, self.basis_gradients_ref)

        # === FIX: Explicitly compute dot product over spatial dimension ===
        # grad_dot[cell, quad, i, j] = ∇φ_i · ∇φ_j  (dot product over dim)
        grad_dot = np.einsum('cqib,cqjb->cqij', grad_phy, grad_phy)
        # Shape: (num_cells, num_quad, num_basis, num_basis)
    
        # === FIX: Correct einsum for quadrature sum ===
        # K_all[cell, i, j] = Σ_q w_q |detJ| (∇φ_i · ∇φ_j)
        K_all = np.einsum('q,c,cqij->cij',
                         self.quadrature_weights,  # (num_quad,)
                         detJ_all,                 # (num_cells,)
                         grad_dot)                 # (num_cells, num_quad, num_basis, num_basis)
        # Result shape: (num_cells, num_basis, num_basis)

        return jnp.array(K_all)
    
    def get_facet_quadrature(self, facet_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get quadrature points and weights for a facet.
        
        Args:
            facet_idx: Global facet index
            
        Returns:
            Tuple of (quad_points_phy, quad_weights, normal)
            quad_points_phy: Quadrature points in physical space, shape (num_quad, dim)
        """
        # Get facet information
        normal = self.mesh.facet_normals[facet_idx]
        area = self.mesh.facet_areas[facet_idx]
        
        # Get reference quadrature
        if self.mesh.dim == 2:
            # For 2D: edge is 1D interval [-1, 1]
            quad_points_ref, quad_weights = basix.make_quadrature(
                basix.CellType.interval,
                self.quadrature_degree
            )
            
            # Map to physical edge
            cell_idx = self.mesh.facet_to_element[facet_idx, 0]
            local_facet = self.mesh.element_to_facet[cell_idx, :]
            local_idx = np.where(local_facet == facet_idx)[0][0]
            facet_vertices = self.mesh.get_facets(self.mesh.cells[cell_idx])[local_idx]
            
            v1 = self.mesh.vertices[facet_vertices[0]]
            v2 = self.mesh.vertices[facet_vertices[1]]
            
            # Ensure quad_points_ref is 1D and then reshape for broadcasting
            if quad_points_ref.ndim > 1:
                quad_points_ref = quad_points_ref.squeeze()
            
            # Linear mapping from [-1, 1] to physical edge
            # quad_points_ref has shape (num_quad,)
            # We need to broadcast to (num_quad, 2)
            quad_points_phy = 0.5 * (v1[None, :] + v2[None, :]) + \
                             0.5 * quad_points_ref[:, None] * (v2 - v1)[None, :]
            
        elif self.mesh.dim == 3:
            # For 3D: face is reference triangle
            quad_points_ref, quad_weights = basix.make_quadrature(
                basix.CellType.triangle,
                self.quadrature_degree
            )
            
            # Map to physical face
            cell_idx = self.mesh.facet_to_element[facet_idx, 0]
            local_facet = self.mesh.element_to_facet[cell_idx, :]
            local_idx = np.where(local_facet == facet_idx)[0][0]
            facet_vertices = self.mesh.get_facets(self.mesh.cells[cell_idx])[local_idx]
            
            v1 = self.mesh.vertices[facet_vertices[0]]
            v2 = self.mesh.vertices[facet_vertices[1]]
            v3 = self.mesh.vertices[facet_vertices[2]]
            
            # Affine mapping from reference triangle
            quad_points_phy = v1[None, :] + \
                             quad_points_ref[:, 0:1] * (v2 - v1)[None, :] + \
                             quad_points_ref[:, 1:2] * (v3 - v1)[None, :]
        else:
            raise ValueError("Unsupported dimension")
        
        return quad_points_phy, quad_weights, normal
    
    def to_jax_arrays(self) -> Dict[str, jnp.ndarray]:
        """Convert element data to JAX arrays."""
        return {
            'basis_values': jnp.array(self.basis_values),
            'basis_gradients_ref': jnp.array(self.basis_gradients_ref),
            'quadrature_weights': jnp.array(self.quadrature_weights),
            'quadrature_points': jnp.array(self.quadrature_points),
            # Note: num_basis_functions is an int, not an array, so we don't convert it
        }

class DGElement(Element):
    """
    DG Element specialized for hyperbolic conservation laws.
    
    This requires different matrices than elliptic problems:
    - Weak differentiation matrices (Drw, Dsw)
    - Lift operator (LIFT)
    - NOT traditional stiffness matrices
    """
    
    def get_weak_differentiation_matrices(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute weak differentiation matrices for DG formulation of Euler equations.
        
        The weak form is:
        ∫_K ∂F/∂x v dx ≈ ∫_K F ∂v/∂x dx - ∫_{∂K} F* v n_x ds
        
        The volume integral ∫_K F ∂v/∂x dx is computed as: weak_derivative_r @ F
        where weak_derivative_r is the weak differentiation matrix in reference r-direction.
        
        For 2D:
        - weak_derivative_r: weak derivative in r-direction
        - weak_derivative_s: weak derivative in s-direction
        
        These are computed as: weak_derivative_r = vandermonde @ grad_vandermonde_r^T @ inv(modal_mass)
        where vandermonde is the Vandermonde matrix.
        
        Returns:
            Tuple of (weak_derivative_r, weak_derivative_s) for 2D, 
            or (weak_derivative_r, weak_derivative_s, weak_derivative_t) for 3D
            Each matrix has shape (num_basis, num_basis)
        """
        # Get Vandermonde matrix and its gradient
        vandermonde = self._get_vandermonde_matrix()
        grad_vandermonde_r, grad_vandermonde_s = self._get_gradient_vandermonde_matrices()
        
        # Compute mass matrix in modal space: modal_mass = vandermonde @ vandermonde^T
        modal_mass = vandermonde @ vandermonde.T
        modal_mass_inv = np.linalg.inv(modal_mass)
        
        # Weak differentiation matrices
        # weak_derivative_r = vandermonde @ grad_vandermonde_r^T @ modal_mass_inv
        weak_derivative_r = vandermonde @ grad_vandermonde_r.T @ modal_mass_inv
        weak_derivative_s = vandermonde @ grad_vandermonde_s.T @ modal_mass_inv
        
        if self.mesh.dim == 3:
            grad_vandermonde_t = self._get_gradient_vandermonde_matrices()[2]
            weak_derivative_t = vandermonde @ grad_vandermonde_t.T @ modal_mass_inv
            return jnp.array(weak_derivative_r), jnp.array(weak_derivative_s), jnp.array(weak_derivative_t)
        
        return jnp.array(weak_derivative_r), jnp.array(weak_derivative_s)
    
    def get_lift_operator(self) -> jnp.ndarray:
        """
        Compute the LIFT operator for surface-to-volume integration.
        
        The LIFT operator maps surface integrals to volume integrals:
        lift_operator @ (surface_term) gives the contribution of the surface integral
        to the volume residual.
        
        For Euler equations:
        ∫_{∂K} (F_numerical - F_interior) · n · v ds = lift_operator @ (face_scale * (F_numerical - F_interior) * normal)
        
        Construction:
        lift_operator = vandermonde @ vandermonde^T @ surface_mass_matrix
        where surface_mass_matrix is the surface mass matrix.
        
        Returns:
            lift_operator of shape (num_basis, num_faces * num_face_nodes)
        """
        # Get Vandermonde matrix
        vandermonde = self._get_vandermonde_matrix()
        
        # Construct surface mass matrix
        surface_mass_matrix = self._construct_surface_mass_matrix()
        
        # lift_operator = vandermonde @ vandermonde^T @ surface_mass_matrix
        lift_operator = vandermonde @ vandermonde.T @ surface_mass_matrix
        
        return jnp.array(lift_operator)
    
    def _construct_surface_mass_matrix(self) -> np.ndarray:
        """
        Construct the surface mass matrix for the LIFT operator.
        
        surface_mass_matrix has shape (num_basis, num_faces * num_face_nodes)
        For each face, we compute the 1D mass matrix on that face.
        
        Returns:
            Surface mass matrix
        """
        num_basis = self.num_basis_functions
        num_faces = len(self.mesh.get_face_indices()[0])  # Number of faces per element
        
        # Get face quadrature and basis evaluations
        face_basis_values = self._get_face_basis_values()  # (num_faces, num_face_quad, num_basis)
        face_quadrature_weights = self._get_face_quadrature_weights()  # (num_faces, num_face_quad)
        
        # For each face, identify which nodes lie on it
        face_node_masks = self._get_face_node_masks()  # (num_faces, num_basis)
        
        surface_mass_matrix = np.zeros((num_basis, num_faces * num_basis))
        
        for face_idx in range(num_faces):
            # Get nodes on this face
            nodes_on_face = np.where(face_node_masks[face_idx])[0]
            num_nodes_on_face = len(nodes_on_face)
            
            if num_nodes_on_face == 0:
                continue
            
            # Extract basis values for nodes on this face
            basis_vals_on_face = face_basis_values[face_idx, :, nodes_on_face]  # (num_face_quad, num_nodes_on_face)
            
            # Compute 1D mass matrix on this face
            # face_mass[i,j] = ∫_face φ_i φ_j ds
            face_mass_matrix = np.einsum('q,qi,qj->ij', 
                                         face_quadrature_weights[face_idx],
                                         basis_vals_on_face,
                                         basis_vals_on_face)
            
            # Invert face mass matrix
            face_mass_inv = np.linalg.inv(face_mass_matrix)
            
            # Place in surface_mass_matrix
            col_start = face_idx * num_basis
            for i, node_i in enumerate(nodes_on_face):
                for j, node_j in enumerate(nodes_on_face):
                    surface_mass_matrix[node_i, col_start + node_j] = face_mass_inv[i, j]
        
        return surface_mass_matrix
    
    def _get_vandermonde_matrix(self) -> np.ndarray:
        """
        Construct Vandermonde matrix.
        
        vandermonde[i,j] = φ_j(x_i) where φ_j is the j-th modal basis function
        and x_i is the i-th nodal point.
        
        For basix elements, we can use the tabulated values at the nodes.
        
        Returns:
            Vandermonde matrix of shape (num_basis, num_basis)
        """
        # Get nodal points (interpolation nodes)
        nodal_points = self._get_nodal_points()
        
        # Tabulate basis at nodal points
        tabulated = self.element.tabulate(0, nodal_points)
        vandermonde = tabulated[0, :, :, 0]  # Shape: (num_nodes, num_basis)
        
        return vandermonde
    
    def _get_gradient_vandermonde_matrices(self) -> Tuple[np.ndarray, ...]:
        """
        Construct gradient Vandermonde matrices for reference coordinates.
        
        grad_vandermonde_r[i,j] = ∂φ_j/∂r(x_i)
        grad_vandermonde_s[i,j] = ∂φ_j/∂s(x_i)
        
        Returns:
            Tuple of gradient Vandermonde matrices
        """
        nodal_points = self._get_nodal_points()
        
        # Tabulate gradients at nodal points
        tabulated = self.element.tabulate(1, nodal_points)
        
        # Extract gradient matrices
        # tabulated[1:dim+1, :, :, 0] has shape (dim, num_nodes, num_basis)
        grad_vandermonde_r = tabulated[1, :, :, 0]  # ∂/∂r
        grad_vandermonde_s = tabulated[2, :, :, 0]  # ∂/∂s
        
        if self.mesh.dim == 3:
            grad_vandermonde_t = tabulated[3, :, :, 0]  # ∂/∂t
            return grad_vandermonde_r, grad_vandermonde_s, grad_vandermonde_t
        
        return grad_vandermonde_r, grad_vandermonde_s
    
    def _get_nodal_points(self) -> np.ndarray:
        """
        Get nodal interpolation points (e.g., Gauss-Lobatto points).
        
        For DG with modal basis, these are typically:
        - Vertices for P1 elements
        - Gauss-Lobatto points for higher order
        
        Returns:
            Nodal points of shape (num_basis, dim)
        # For simplicity, use equispaced points
        # In practice, you'd use Gauss-Lobatto or other optimal points
        
        if self.mesh.dim == 2:
            # Triangle: use Warp & Blend points or equispaced
            # This is a simplified version - use proper nodal points
            polynomial_degree = self.degree
            num_nodes_on_triangle = (polynomial_degree + 1) * (polynomial_degree + 2) // 2
            
            # Equispaced points on reference triangle
            points = []
            for i in range(polynomial_degree + 1):
                for j in range(polynomial_degree + 1 - i):
                    ref_coord_r = -1 + 2*i/polynomial_degree if polynomial_degree > 0 else -1/3
                    ref_coord_s = -1 + 2*j/polynomial_degree if polynomial_degree > 0 else -1/3
                    points.append([ref_coord_r, ref_coord_s])
            
            return np.array(points[:num_nodes_on_triangle])
        
        elif self.mesh.dim == 3:
            # Tetrahedron
            polynomial_degree = self.degree
            num_nodes_on_tet = (polynomial_degree + 1) * (polynomial_degree + 2) * (polynomial_degree + 3) // 6
            
            points = []
            for i in range(polynomial_degree + 1):
                for j in range(polynomial_degree + 1 - i):
                    for k in range(polynomial_degree + 1 - i - j):
                        ref_coord_r = -1 + 2*i/polynomial_degree if polynomial_degree > 0 else -1/4
                        ref_coord_s = -1 + 2*j/polynomial_degree if polynomial_degree > 0 else -1/4
                        ref_coord_t = -1 + 2*k/polynomial_degree if polynomial_degree > 0 else -1/4
                        points.append([ref_coord_r, ref_coord_s, ref_coord_t])
            
            return np.array(points[:num_nodes_on_tet])
    
        """
        
        # Place holder - using basix nodal points
        return jnp.array([])
    
    def _get_face_basis_values(self) -> np.ndarray:
        """
        Get basis function values at face quadrature points.
        
        Returns:
            Array of shape (num_faces, num_face_quad_points, num_basis)
        # This would use basix's face quadrature
        # Placeholder implementation
        face_indices = self.mesh.get_face_indices()
        num_faces = len(face_indices[0])
        
        # Get face quadrature points (in reference coordinates)
        face_quadrature_points = self._get_face_quadrature_points()
        
        face_basis_values = []
        for face_idx in range(num_faces):
            # Tabulate basis at face quadrature points
            tabulated = self.element.tabulate(0, face_quadrature_points[face_idx])
            values = tabulated[0, :, :, 0]  # (num_face_quad_points, num_basis)
            face_basis_values.append(values)
        
        return np.array(face_basis_values)
    
        """
        # Placeholder - would use proper face basis evaluations
        return np.array([])
    
    def _get_face_quadrature_points(self) -> list:
        """Get quadrature points on each face."""
        # Placeholder - would use proper face quadrature
        return np.array([]) 
    
    def _get_face_quadrature_weights(self) -> np.ndarray:
        """Get quadrature weights for each face."""
        # Placeholder
        return np.array([])
    
    def _get_face_node_masks(self) -> np.ndarray:
        """
        Get boolean masks indicating which nodes lie on each face.
        
        Returns:
            Array of shape (num_faces, num_basis)
        """
        # This depends on the element type and nodal point distribution
        # Placeholder
        num_faces = 3  # For triangle
        return np.zeros((num_faces, self.num_basis_functions), dtype=bool)


# Example usage showing how these matrices are used in Euler equations
def euler_dg_rhs_example(element: DGElement, 
                         conservative_vars: np.ndarray,
                         flux_function) -> np.ndarray:
    """
    Example of computing RHS for Euler equations using DG matrices.
    
    This mirrors the structure in your Euler2D.py code.
    
    Args:
        element: DG element with weak operators
        conservative_vars: Conservative variables (num_cells, num_basis, num_vars)
        flux_function: Function to compute fluxes flux_x, flux_y from conservative_vars
    
    Returns:
        RHS of shape (num_cells, num_basis, num_vars)
    """
    # Get weak differentiation matrices
    weak_derivative_r, weak_derivative_s = element.get_weak_differentiation_matrices()
    
    # Get LIFT operator
    lift_operator = element.get_lift_operator()
    
    # Get geometric factors (from your code)
    metric_rx, metric_sx, metric_ry, metric_sy, jacobian = element.compute_geometric_factors()
    
    # Compute fluxes
    flux_x, flux_y = flux_function(conservative_vars)  # (num_cells, num_basis, num_vars)
    
    # Volume integrals (weak form)
    num_variables = conservative_vars.shape[-1]
    rhs_conservative = np.zeros_like(conservative_vars)
    
    for var_idx in range(num_variables):
        # Weak derivatives in reference coordinates
        dflux_x_dr = weak_derivative_r @ flux_x[:, :, var_idx]  # Weak derivative in r
        dflux_x_ds = weak_derivative_s @ flux_x[:, :, var_idx]  # Weak derivative in s
        dflux_y_dr = weak_derivative_r @ flux_y[:, :, var_idx]
        dflux_y_ds = weak_derivative_s @ flux_y[:, :, var_idx]
        
        # Transform to physical coordinates using metric terms
        rhs_conservative[:, :, var_idx] = (metric_rx * dflux_x_dr + 
                                           metric_sx * dflux_x_ds + 
                                           metric_ry * dflux_y_dr + 
                                           metric_sy * dflux_y_ds)
    
    # Surface integrals (numerical flux)
    # This would compute F_numerical at faces and apply lift_operator
    # surface_contribution = lift_operator @ (face_scale * numerical_flux)
    # rhs_conservative -= surface_contribution
    
    return rhs_conservative