# Element.py
import abc
import basix
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
import pdb
from Mesh import DGMesh, TriangularMesh, TetrahedralMesh

class DGElement(abc.ABC):
    """
    Abstract base class for Discontinuous Galerkin finite elements.
    Provides core functionality for DG methods with basix integration.
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


class DGTriangularElement(DGElement):
    """DG element on triangular mesh."""
    
    def create_element(self):
        """Create discontinuous Lagrange element on triangle."""
        self.element = basix.create_element(
            basix.ElementFamily.P,
            basix.CellType.triangle,
            self.degree,
            basix.LagrangeVariant.equispaced,
            discontinuous=True
        )
    
    def create_quadrature(self):
        """Create quadrature rule on reference triangle."""
        self.quadrature_points, self.quadrature_weights = basix.make_quadrature(
            basix.CellType.triangle,
            self.quadrature_degree
        )


class DGTetrahedralElement(DGElement):
    """DG element on tetrahedral mesh."""
    
    def create_element(self):
        """Create discontinuous Lagrange element on tetrahedron."""
        self.element = basix.create_element(
            basix.ElementFamily.P,
            basix.CellType.tetrahedron,
            self.degree,
            basix.LagrangeVariant.equispaced,
            discontinuous=True
        )
    
    def create_quadrature(self):
        """Create quadrature rule on reference tetrahedron."""
        self.quadrature_points, self.quadrature_weights = basix.make_quadrature(
            basix.CellType.tetrahedron,
            self.quadrature_degree
        )


class ElementFactory:
    """Factory for creating DG elements based on mesh type."""
    
    @staticmethod
    def create_element(mesh: DGMesh, degree: int, quadrature_degree: Optional[int] = None) -> DGElement:
        """
        Create appropriate DG element for the given mesh.
        
        Args:
            mesh: DGMesh instance
            degree: Polynomial degree
            quadrature_degree: Quadrature degree (optional)
            
        Returns:
            DGElement instance
        """
        if isinstance(mesh, TriangularMesh):
            return DGTriangularElement(mesh, degree, quadrature_degree)
        elif isinstance(mesh, TetrahedralMesh):
            return DGTetrahedralElement(mesh, degree, quadrature_degree)
        else:
            raise ValueError(f"Unsupported mesh type: {type(mesh).__name__}")


# ============================================================================
# UNIT TESTS
# ============================================================================

import unittest

class TestDGElement(unittest.TestCase):
    """Unit tests for DGElement classes."""
    
    def setUp(self):
        """Set up test meshes and elements."""
        from Mesh import MeshFactory
        
        self.tri_mesh = MeshFactory.create_triangular_cross_mesh()
        self.tet_mesh = MeshFactory.create_single_tetrahedron()
        
        self.tri_element = ElementFactory.create_element(self.tri_mesh, degree=2)
        self.tet_element = ElementFactory.create_element(self.tet_mesh, degree=1)
    
    def test_triangular_element_creation(self):
        """Test creation of triangular DG element."""
        self.assertIsInstance(self.tri_element, DGTriangularElement)
        self.assertIsInstance(self.tri_element, DGElement)
        self.assertEqual(self.tri_element.degree, 2)
        self.assertEqual(self.tri_element.mesh, self.tri_mesh)
    
    def test_tetrahedral_element_creation(self):
        """Test creation of tetrahedral DG element."""
        self.assertIsInstance(self.tet_element, DGTetrahedralElement)
        self.assertIsInstance(self.tet_element, DGElement)
        self.assertEqual(self.tet_element.degree, 1)
        self.assertEqual(self.tet_element.mesh, self.tet_mesh)
    
    def test_element_properties(self):
        """Test element properties like basis function count."""
        # For P2 triangle: 6 basis functions
        self.assertEqual(self.tri_element.num_basis_functions, 6)
        
        # For P1 tetrahedron: 4 basis functions
        self.assertEqual(self.tet_element.num_basis_functions, 4)
    
    def test_tabulated_basis_shapes(self):
        """Test shapes of tabulated basis functions."""
        # Check basis values shape
        num_quad = len(self.tri_element.quadrature_weights)
        num_basis = self.tri_element.num_basis_functions
        
        self.assertEqual(self.tri_element.basis_values.shape, (num_quad, num_basis))
        
        # Check gradients shape
        self.assertEqual(self.tri_element.basis_gradients_ref.shape, 
                        (num_quad, num_basis, self.tri_mesh.dim))
    
    def test_jacobian_computation_batch(self):
        """Test Jacobian computation for all elements at once."""
        J_all, detJ_all = self.tri_element.compute_element_jacobians()
        
        # Check shapes
        num_cells = len(self.tri_mesh.cells)
        self.assertEqual(J_all.shape, (num_cells, self.tri_mesh.dim, self.tri_mesh.dim))
        self.assertEqual(detJ_all.shape, (num_cells,))
        
        # Check all determinants are positive
        self.assertTrue(np.all(detJ_all > 0))
        
        # For a well-formed mesh, determinants shouldn't be too small
        self.assertTrue(np.all(detJ_all > 1e-10))
    
    def test_mass_matrix_shape(self):
        """Test mass matrix shape and properties."""
        mass_mat = self.tri_element.get_mass_matrices()
        
        # Check shape
        num_cells = len(self.tri_mesh.cells)
        num_basis = self.tri_element.num_basis_functions
        
        self.assertEqual(mass_mat.shape, (num_cells, num_basis, num_basis))
        
        # Check symmetry for each element (with reasonable tolerance)
        for cell_idx in range(num_cells):
            M = mass_mat[cell_idx]
            np.testing.assert_allclose(M, M.T, rtol=1e-10, atol=1e-12)
        
        # Check positive definiteness (eigenvalues > 0)
        for cell_idx in range(num_cells):
            M = mass_mat[cell_idx]
            eigvals = np.linalg.eigvals(M)
            self.assertTrue(np.all(eigvals > 0))
    
    def test_stiffness_matrix_shape(self):
        """Test stiffness matrix shape and properties."""
        stiffness_mat = self.tri_element.get_stiffness_matrices()
        
        # Check shape
        num_cells = len(self.tri_mesh.cells)
        num_basis = self.tri_element.num_basis_functions
        
        self.assertEqual(stiffness_mat.shape, (num_cells, num_basis, num_basis))
        
        # Check symmetry for each element
        for cell_idx in range(num_cells):
            K = stiffness_mat[cell_idx]
            np.testing.assert_allclose(K, K.T, rtol=1e-10, atol=1e-12)
        
        # Check positive semi-definiteness (eigenvalues >= 0)
        for cell_idx in range(num_cells):
            K = stiffness_mat[cell_idx]
            eigvals = np.linalg.eigvals(K)
            print(eigvals)
            self.assertTrue(np.all(eigvals >= -1e-10))
    
    def test_jax_conversion_element(self):
        """Test conversion to JAX arrays."""
        jax_data = self.tri_element.to_jax_arrays()
        
        # Check all required keys
        required_keys = {
            'basis_values', 'basis_gradients_ref', 'quadrature_weights',
            'quadrature_points'
        }
        
        self.assertTrue(required_keys.issubset(jax_data.keys()))
        
        # Check types (all should be JAX arrays)
        for key, value in jax_data.items():
            self.assertIsInstance(value, jnp.ndarray, f"{key} should be JAX array")
        
        # Check that non-array attributes are not included
        self.assertNotIn('num_basis_functions', jax_data)
    
    def test_facet_quadrature_triangular(self):
        """Test facet quadrature for triangular mesh."""
        mesh = self.tri_mesh
        
        # Get a boundary facet
        if len(mesh.boundary_facets) > 0:
            facet_idx = mesh.boundary_facets[0]
            quad_points, quad_weights, normal = self.tri_element.get_facet_quadrature(facet_idx)
            
            # Check shapes
            self.assertEqual(quad_points.shape[1], mesh.dim)
            self.assertEqual(len(quad_weights), len(quad_points))
            self.assertEqual(normal.shape, (mesh.dim,))
            
            # Check that normal is unit length
            normal_norm = np.linalg.norm(normal)
            np.testing.assert_allclose(normal_norm, 1.0, rtol=1e-10)
    
    def test_facet_quadrature_tetrahedral(self):
        """Test facet quadrature for tetrahedral mesh."""
        mesh = self.tet_mesh
        
        # Get a boundary facet
        if len(mesh.boundary_facets) > 0:
            facet_idx = mesh.boundary_facets[0]
            quad_points, quad_weights, normal = self.tet_element.get_facet_quadrature(facet_idx)
            
            # Check shapes
            self.assertEqual(quad_points.shape[1], mesh.dim)
            self.assertEqual(len(quad_weights), len(quad_points))
            self.assertEqual(normal.shape, (mesh.dim,))
            
            # Check that normal is unit length
            normal_norm = np.linalg.norm(normal)
            np.testing.assert_allclose(normal_norm, 1.0, rtol=1e-10)


class TestElementFactory(unittest.TestCase):
    """Test element factory functionality."""
    
    def test_factory_triangular(self):
        """Test factory creation for triangular mesh."""
        from Mesh import MeshFactory
        
        tri_mesh = MeshFactory.create_triangular_cross_mesh()
        element = ElementFactory.create_element(tri_mesh, degree=3)
        
        self.assertIsInstance(element, DGTriangularElement)
        self.assertEqual(element.degree, 3)
        self.assertEqual(element.num_basis_functions, 10)  # P3 triangle has 10 basis functions
    
    def test_factory_tetrahedral(self):
        """Test factory creation for tetrahedral mesh."""
        from Mesh import MeshFactory
        
        tet_mesh = MeshFactory.create_single_tetrahedron()
        element = ElementFactory.create_element(tet_mesh, degree=2)
        
        self.assertIsInstance(element, DGTetrahedralElement)
        self.assertEqual(element.degree, 2)
        self.assertEqual(element.num_basis_functions, 10)  # P2 tetrahedron has 10 basis functions


def run_element_tests():
    """Run all element tests."""
    print("="*80)
    print("RUNNING ELEMENT.PY TEST SUITE")
    print("="*80)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestDGElement))
    suite.addTests(loader.loadTestsFromTestCase(TestElementFactory))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    print("ELEMENT TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


def demonstrate_element_usage():
    """Demonstrate practical usage of DGElement."""
    from Mesh import MeshFactory
    
    print("\n" + "="*80)
    print("DG ELEMENT USAGE DEMONSTRATION")
    print("="*80)
    
    # Example 1: Create element on triangular mesh
    print("\n1. Triangular DG Element (P2):")
    tri_mesh = MeshFactory.create_triangular_cross_mesh()
    tri_element = ElementFactory.create_element(tri_mesh, degree=2)
    
    print(f"   - Mesh: {len(tri_mesh.cells)} elements")
    print(f"   - Element degree: {tri_element.degree}")
    print(f"   - Basis functions: {tri_element.num_basis_functions}")
    print(f"   - Quadrature points: {len(tri_element.quadrature_weights)}")
    
    # Compute mass matrix
    mass_mat = tri_element.get_mass_matrices()
    print(f"   - Mass matrix shape: {mass_mat.shape}")
    print(f"   - Average element mass: {np.mean(mass_mat):.6f}")
    
    # Compute stiffness matrix
    stiffness_mat = tri_element.get_stiffness_matrices()
    print(f"   - Stiffness matrix shape: {stiffness_mat.shape}")
    
    # Example 2: Create element on tetrahedral mesh
    print("\n2. Tetrahedral DG Element (P1):")
    tet_mesh = MeshFactory.create_single_tetrahedron()
    tet_element = ElementFactory.create_element(tet_mesh, degree=1)
    
    print(f"   - Mesh: {len(tet_mesh.cells)} elements")
    print(f"   - Element degree: {tet_element.degree}")
    print(f"   - Basis functions: {tet_element.num_basis_functions}")
    
    # Example 3: Facet quadrature for flux terms
    print("\n3. Facet Quadrature (for DG flux terms):")
    if len(tri_mesh.boundary_facets) > 0:
        facet_idx = tri_mesh.boundary_facets[0]
        quad_points, quad_weights, normal = tri_element.get_facet_quadrature(facet_idx)
        
        print(f"   - Boundary facet {facet_idx}:")
        print(f"     Quadrature points: {len(quad_points)}")
        print(f"     Normal vector: {normal}")
        print(f"     Area: {tri_mesh.facet_areas[facet_idx]:.6f}")
        
        # Test that quadrature integrates correctly on reference facet
        # ∫ 1 dΩ should equal area
        computed_area = np.sum(quad_weights)
        print(f"     Computed area from quadrature: {computed_area:.6f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run tests
    success = run_element_tests()
    
    # Run demonstration if tests pass
    if success:
        demonstrate_element_usage()
    
    # Exit
    import sys
    sys.exit(0 if success else 1)