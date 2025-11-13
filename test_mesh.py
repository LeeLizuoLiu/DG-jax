
class MeshConnectivityValidator:
    """Comprehensive validator for DG mesh connectivity."""
    
    def __init__(self, mesh: DGMesh):
        self.mesh = mesh
        self.results: Dict[str, bool] = {}
        self.messages: Dict[str, str] = {}
    
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all validation tests and return results.
        
        Returns:
            Dictionary mapping test names to pass/fail status
        """
        tests = [
            ('Vertex Indices Validity', self._test_vertex_indices),
            ('Element-to-Vertex Consistency', self._test_element_to_vertex),
            ('Element-to-Element Symmetry', self._test_element_symmetry),
            ('Facet Consistency', self._test_facet_consistency),
            ('Boundary Facets Identification', self._test_boundary_facets),
            ('Mesh Connectivity', self._test_mesh_connectivity),
            ('Facet Sharing Correctness', self._test_facet_sharing),
            ('Left-Right Mapping', self._test_left_right_mapping),
            ('DG Geometry Data', self._test_dg_geometry),
        ]
        
        for test_name, test_func in tests:
            try:
                passed, message = test_func()
                self.results[test_name] = passed
                self.messages[test_name] = message
            except Exception as e:
                self.results[test_name] = False
                self.messages[test_name] = f"Exception: {str(e)}"
        
        return self.results
    
    def print_results(self):
        """Print validation results in a formatted manner."""
        print("\n" + "="*60)
        print("MESH VALIDATION RESULTS")
        print("="*60)
        
        for test_name, passed in self.results.items():
            status_symbol = "✓" if passed else "✗"
            message = self.messages[test_name]
            print(f"{status_symbol} {test_name}: {message}")
        
        total_passed = sum(self.results.values())
        total_tests = len(self.results)
        print(f"\nSummary: {total_passed}/{total_tests} tests passed")
        
        if total_passed == total_tests:
            print("🎉 All tests passed! Mesh is DG-ready.")
        else:
            print("⚠️  Some tests failed. Review mesh construction.")
        
        print("="*60)
    
    def _test_vertex_indices(self) -> Tuple[bool, str]:
        """Test that all vertex indices are within valid range."""
        max_vertex = len(self.mesh.vertices) - 1
        
        for cell_idx, cell in enumerate(self.mesh.cells):
            if np.any(cell < 0) or np.any(cell > max_vertex):
                invalid = cell[(cell < 0) | (cell > max_vertex)]
                return False, f"Cell {cell_idx} has invalid vertex indices: {invalid}"
        
        return True, "All vertex indices are valid"
    
    def _test_element_to_vertex(self) -> Tuple[bool, str]:
        """Test consistency between cells and element_to_vertex."""
        if not np.array_equal(self.mesh.cells, self.mesh.element_to_vertex):
            return False, "element_to_vertex does not match cells"
        return True, "element_to_vertex is consistent with cells"
    
    def _test_element_symmetry(self) -> Tuple[bool, str]:
        """Test that element-to-element neighbor relationships are symmetric."""
        for cell_idx in range(len(self.mesh.cells)):
            for local_facet, neighbor in enumerate(self.mesh.element_to_element[cell_idx]):
                if neighbor != -1:
                    # Check if neighbor points back to this cell
                    if cell_idx not in self.mesh.element_to_element[neighbor]:
                        return False, f"Asymmetric: {cell_idx}->{neighbor} but not reciprocated"
        
        return True, "All neighbor relationships are symmetric"
    
    def _test_facet_consistency(self) -> Tuple[bool, str]:
        """Test consistency between element_to_facet and facet_to_element."""
        for cell_idx in range(len(self.mesh.cells)):
            for local_facet, facet_idx in enumerate(self.mesh.element_to_facet[cell_idx]):
                if facet_idx != -1:
                    if cell_idx not in self.mesh.facet_to_element[facet_idx]:
                        return False, f"Facet {facet_idx} does not reference cell {cell_idx}"
        
        return True, "Facet connectivity is consistent"
    
    def _test_boundary_facets(self) -> Tuple[bool, str]:
        """Test correct identification of boundary facets."""
        boundary_count = 0
        for facet_idx in range(len(self.mesh.facet_to_element)):
            cell_count = np.sum(self.mesh.facet_to_element[facet_idx] != -1)
            
            if cell_count == 1:
                boundary_count += 1
                if facet_idx not in self.mesh.boundary_facets:
                    return False, f"Facet {facet_idx} is boundary but not marked"
            elif cell_count == 2:
                if facet_idx in self.mesh.boundary_facets:
                    return False, f"Facet {facet_idx} is interior but marked as boundary"
        
        if len(self.mesh.boundary_facets) != boundary_count:
            return False, f"Boundary count mismatch: {len(self.mesh.boundary_facets)} vs {boundary_count}"
        
        return True, f"Correctly identified {boundary_count} boundary facets"
    
    def _test_mesh_connectivity(self) -> Tuple[bool, str]:
        """Test that the mesh is fully connected (no isolated components)."""
        visited: Set[int] = set()
        stack = [0]  # Start from first element
        
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                neighbors = [n for n in self.mesh.element_to_element[current] if n != -1]
                stack.extend(neighbors)
        
        if len(visited) != len(self.mesh.cells):
            return False, f"Mesh not fully connected: {len(visited)}/{len(self.mesh.cells)} elements reachable"
        
        return True, "Mesh is fully connected"
    
    def _test_facet_sharing(self) -> Tuple[bool, str]:
        """Test that facets are shared by at most 2 elements."""
        facet_usage: Dict[Tuple[int, ...], List[int]] = {}
        
        for cell_idx in range(len(self.mesh.cells)):
            facets = self.mesh.get_facets(self.mesh.cells[cell_idx])
            for facet in facets:
                sorted_facet = tuple(sorted(facet))
                facet_usage.setdefault(sorted_facet, []).append(cell_idx)
        
        for facet, cells in facet_usage.items():
            if len(cells) > 2:
                return False, f"Facet {facet} shared by {len(cells)} elements"
            
            if len(cells) == 2:
                # Verify they are neighbors
                cell1, cell2 = cells
                is_neighbor = (
                    cell2 in self.mesh.element_to_element[cell1] or
                    cell1 in self.mesh.element_to_element[cell2]
                )
                if not is_neighbor:
                    return False, f"Facet {facet} shared but elements are not neighbors"
        
        return True, f"All {len(facet_usage)} facets correctly shared (max 2 elements)"
    
    def _test_left_right_mapping(self) -> Tuple[bool, str]:
        """Test consistency of left-right cell mapping."""
        if self.mesh.left_right_cells is None:
            return False, "Left-right mapping not initialized"
        
        for facet_idx, (left_cell, right_cell) in enumerate(self.mesh.left_right_cells):
            # Check boundary facets
            if facet_idx in self.mesh.boundary_facets:
                if right_cell != -1:
                    return False, f"Boundary facet {facet_idx} has non-null right cell"
                if left_cell not in self.mesh.facet_to_element[facet_idx]:
                    return False, f"Left cell {left_cell} not in facet_to_element for boundary facet {facet_idx}"
            else:
                # Check interior facets
                if right_cell == -1:
                    return False, f"Interior facet {facet_idx} has null right cell"
                if left_cell >= right_cell:
                    return False, f"Left cell {left_cell} >= right cell {right_cell} for facet {facet_idx}"
                if left_cell not in self.mesh.facet_to_element[facet_idx] or right_cell not in self.mesh.facet_to_element[facet_idx]:
                    return False, f"Left/right cells not consistent with facet_to_element for facet {facet_idx}"
        
        return True, "Left-right mapping is consistent"
    
    def _test_dg_geometry(self) -> Tuple[bool, str]:
        """Test DG geometry data (normals, areas, centroids)."""
        if self.mesh.facet_normals is None or self.mesh.facet_areas is None or self.mesh.facet_centroids is None:
            return False, "DG geometry data not initialized"
        
        # Check shapes
        num_facets = len(self.mesh.facet_to_element)
        if self.mesh.facet_normals.shape != (num_facets, self.mesh.dim):
            return False, f"Incorrect facet_normals shape: {self.mesh.facet_normals.shape}"
        if self.mesh.facet_areas.shape != (num_facets,):
            return False, f"Incorrect facet_areas shape: {self.mesh.facet_areas.shape}"
        if self.mesh.facet_centroids.shape != (num_facets, self.mesh.dim):
            return False, f"Incorrect facet_centroids shape: {self.mesh.facet_centroids.shape}"
        
        # Check for non-zero areas
        zero_area_facets = np.where(np.abs(self.mesh.facet_areas) < 1e-12)[0]
        if len(zero_area_facets) > 0:
            return False, f"Facets with zero area: {zero_area_facets}"
        
        # Check for unit normals
        for facet_idx in range(num_facets):
            normal = self.mesh.facet_normals[facet_idx]
            norm_length = np.linalg.norm(normal)
            if abs(norm_length - 1.0) > 1e-6:
                return False, f"Facet {facet_idx} normal is not unit length: {norm_length}"
        
        return True, "DG geometry data is valid"


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestDGMesh(unittest.TestCase):
    """Comprehensive unit tests for DG mesh classes."""
    
    def setUp(self):
        """Set up test meshes."""
        self.cross_mesh = MeshFactory.create_triangular_cross_mesh()
        self.unit_square_mesh = MeshFactory.create_triangular_unit_square(3, 3)
        self.single_tet = MeshFactory.create_single_tetrahedron()
        self.two_tet = MeshFactory.create_two_tetrahedra()
    
    def test_triangular_mesh_inheritance(self):
        """Test that TriangularMesh properly inherits from DGMesh."""
        self.assertIsInstance(self.cross_mesh, DGMesh)
        self.assertIsInstance(self.cross_mesh, TriangularMesh)
        
        self.assertEqual(self.cross_mesh.dim, 2)
        self.assertEqual(self.cross_mesh.get_facet_type(), "edge")
        self.assertEqual(self.cross_mesh.num_facets_per_element(), 3)
    
    def test_tetrahedral_mesh_inheritance(self):
        """Test that TetrahedralMesh properly inherits from DGMesh."""
        self.assertIsInstance(self.single_tet, DGMesh)
        self.assertIsInstance(self.single_tet, TetrahedralMesh)
        
        self.assertEqual(self.single_tet.dim, 3)
        self.assertEqual(self.single_tet.get_facet_type(), "face")
        self.assertEqual(self.single_tet.num_facets_per_element(), 4)
    
    def test_triangular_connectivity_shapes(self):
        """Test correct array shapes for triangular mesh connectivity."""
        mesh = self.cross_mesh
        
        # Check element_to_vertex matches cells
        self.assertEqual(mesh.element_to_vertex.shape, mesh.cells.shape)
        
        # Check element_to_element shape
        self.assertEqual(mesh.element_to_element.shape, (len(mesh.cells), 3))
        
        # Check element_to_facet shape
        self.assertEqual(mesh.element_to_facet.shape, (len(mesh.cells), 3))
        
        # Check facet_to_element shape
        num_facets = len(mesh.facet_to_element)
        self.assertEqual(mesh.facet_to_element.shape, (num_facets, 2))
        
        # Check boundary facets is a list
        self.assertIsInstance(mesh.boundary_facets, list)
    
    def test_tetrahedral_connectivity_shapes(self):
        """Test correct array shapes for tetrahedral mesh connectivity."""
        mesh = self.single_tet
        
        # Single tetrahedron should have 4 boundary faces
        self.assertEqual(len(mesh.boundary_facets), 4)
        
        # element_to_element should be (1, 4)
        self.assertEqual(mesh.element_to_element.shape, (1, 4))
        
        # facet_to_element should have 4 rows (one per face)
        self.assertEqual(mesh.facet_to_element.shape, (4, 2))
    
    def test_dg_geometry_shapes(self):
        """Test correct shapes of DG geometry arrays."""
        mesh = self.cross_mesh
        
        num_facets = len(mesh.facet_to_element)
        
        # Check facet normals shape
        self.assertEqual(mesh.facet_normals.shape, (num_facets, mesh.dim))
        
        # Check facet areas shape
        self.assertEqual(mesh.facet_areas.shape, (num_facets,))
        
        # Check facet centroids shape
        self.assertEqual(mesh.facet_centroids.shape, (num_facets, mesh.dim))
        
        # Check left-right cells shape
        self.assertEqual(mesh.left_right_cells.shape, (num_facets, 2))
        
        # Check boundary types shape
        self.assertEqual(mesh.facet_boundary_types.shape, (num_facets,))
    
    def test_left_right_mapping_boundary(self):
        """Test left-right mapping for boundary facets."""
        mesh = self.single_tet
        
        for facet_idx in mesh.boundary_facets:
            left, right = mesh.left_right_cells[facet_idx]
            self.assertNotEqual(left, -1)  # Left cell should exist
            self.assertEqual(right, -1)    # Right cell should be -1 for boundary
    
    def test_left_right_mapping_interior(self):
        """Test left-right mapping for interior facets."""
        mesh = self.two_tet
        
        interior_facets = set(range(len(mesh.facet_to_element))) - set(mesh.boundary_facets)
        
        self.assertTrue(len(interior_facets) > 0, "Should have at least one interior facet")
        
        for facet_idx in interior_facets:
            left, right = mesh.left_right_cells[facet_idx]
            self.assertNotEqual(left, -1)
            self.assertNotEqual(right, -1)
            self.assertLess(left, right, "Left cell index must be less than right")
    
    def test_boundary_condition_marking(self):
        """Test manual boundary condition marking."""
        mesh = self.cross_mesh
        
        # Initially, all boundary facets should be marked as Dirichlet (1)
        for facet_idx in mesh.boundary_facets:
            self.assertEqual(mesh.facet_boundary_types[facet_idx], 1)
        
        # Change some boundary facets to Neumann
        neumann_facets = mesh.boundary_facets[:2]
        mesh.mark_boundary_condition(neumann_facets, 2)
        
        for facet_idx in neumann_facets:
            self.assertEqual(mesh.facet_boundary_types[facet_idx], 2)
        
        # Verify other boundaries remain Dirichlet
        for facet_idx in mesh.boundary_facets[2:]:
            self.assertEqual(mesh.facet_boundary_types[facet_idx], 1)
    
    def test_invalid_boundary_marking_raises_error(self):
        """Test that marking interior facets as boundaries raises an error."""
        mesh = self.cross_mesh
        
        interior_facets = [i for i in range(len(mesh.facet_to_element)) 
                          if i not in mesh.boundary_facets]
        
        if interior_facets:
            with self.assertRaises(ValueError):
                mesh.mark_boundary_condition([interior_facets[0]], 2)
    
    def test_jax_conversion(self):
        """Test conversion of mesh data to JAX arrays."""
        mesh = self.cross_mesh
        
        jax_data = mesh.to_jax_arrays()
        
        # Check that all required keys are present
        required_keys = {
            'vertices', 'cells', 'element_to_element', 'element_to_facet',
            'facet_to_element', 'facet_normals', 'facet_areas', 'facet_centroids',
            'left_right_cells', 'facet_boundary_types'
        }
        
        self.assertTrue(required_keys.issubset(jax_data.keys()))
        
        # Check that all values are JAX arrays
        for key, value in jax_data.items():
            self.assertIsInstance(value, jnp.ndarray, f"{key} should be a JAX array")
    
    def test_mesh_statistics(self):
        """Test mesh statistics computation."""
        mesh = self.unit_square_mesh
        
        stats = mesh.get_mesh_statistics()
        
        # Check required keys
        required_keys = {'num_vertices', 'num_cells', 'num_facets', 'num_boundary_facets', 
                        'dimension', 'facet_type', 'elements_per_cell', 
                        'avg_neighbors_per_cell', 'min_neighbors', 'max_neighbors'}
        
        self.assertTrue(required_keys.issubset(stats.keys()))
        
        # Check consistency
        self.assertEqual(stats['num_vertices'], len(mesh.vertices))
        self.assertEqual(stats['num_cells'], len(mesh.cells))
        self.assertEqual(stats['num_facets'], len(mesh.facet_to_element))
        self.assertEqual(stats['num_boundary_facets'], len(mesh.boundary_facets))
        
        # Check geometry stats
        self.assertIn('min_facet_area', stats)
        self.assertIn('max_facet_area', stats)
        self.assertGreater(stats['total_boundary_area'], 0.0)
    
    def test_single_element_mesh(self):
        """Test mesh with a single element (all boundaries)."""
        # Single triangle
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = np.array([[0, 1, 2]], dtype=int)
        single_tri = TriangularMesh(vertices, cells)
        
        self.assertEqual(len(single_tri.boundary_facets), 3)
        self.assertTrue(np.all(single_tri.element_to_element[0] == -1))
        
        # Single tetrahedron
        single_tet = MeshFactory.create_single_tetrahedron()
        self.assertEqual(len(single_tet.boundary_facets), 4)
    
    def test_validation_passes_on_good_mesh(self):
        """Test that validation passes on correctly constructed meshes."""
        meshes = [
            self.cross_mesh,
            self.unit_square_mesh,
            self.single_tet,
            self.two_tet
        ]
        
        for mesh in meshes:
            with self.subTest(mesh_type=type(mesh).__name__):
                validator = MeshConnectivityValidator(mesh)
                results = validator.run_all_tests()
                
                # All tests should pass
                self.assertTrue(all(results.values()), 
                              f"Mesh validation failed: {results}")
    
    def test_facet_geometry_correctness(self):
        """Test correctness of facet geometry calculations."""
        # Test triangular mesh edge lengths
        mesh = self.cross_mesh
        
        # For the cross mesh, boundary edges should have known lengths
        for facet_idx in mesh.boundary_facets:
            area = mesh.facet_areas[facet_idx]
            # Edge lengths should be sqrt(2)/2 or 1.0
            self.assertTrue(np.isclose(area, 0.5 * np.sqrt(2)) or np.isclose(area, 1.0))
    
    def test_unit_normal_vectors(self):
        """Test that facet normals are unit vectors."""
        mesh = self.cross_mesh
        
        for facet_idx in range(len(mesh.facet_to_element)):
            normal = mesh.facet_normals[facet_idx]
            norm = np.linalg.norm(normal)
            self.assertAlmostEqual(norm, 1.0, places=6, 
                                 msg=f"Facet {facet_idx} normal is not unit length: {norm}")


class TestMeshFactory(unittest.TestCase):
    """Test the mesh factory functionality."""
    
    def test_create_triangular_cross_mesh(self):
        """Test creation of cross mesh."""
        mesh = MeshFactory.create_triangular_cross_mesh()
        
        self.assertIsInstance(mesh, TriangularMesh)
        self.assertEqual(len(mesh.vertices), 5)
        self.assertEqual(len(mesh.cells), 4)
    
    def test_create_unit_square_mesh(self):
        """Test creation of structured unit square mesh."""
        mesh = MeshFactory.create_triangular_unit_square(2, 2)
        
        self.assertIsInstance(mesh, TriangularMesh)
        self.assertEqual(len(mesh.vertices), 9)  # (2+1)*(2+1)
        self.assertEqual(len(mesh.cells), 8)     # 2*2*2
    
    def test_create_single_tetrahedron(self):
        """Test creation of single tetrahedron."""
        mesh = MeshFactory.create_single_tetrahedron()
        
        self.assertIsInstance(mesh, TetrahedralMesh)
        self.assertEqual(len(mesh.vertices), 4)
        self.assertEqual(len(mesh.cells), 1)
    
    def test_create_two_tetrahedra(self):
        """Test creation of two tetrahedra mesh."""
        mesh = MeshFactory.create_two_tetrahedra()
        
        self.assertIsInstance(mesh, TetrahedralMesh)
        self.assertEqual(len(mesh.vertices), 5)
        self.assertEqual(len(mesh.cells), 2)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_comprehensive_tests():
    """
    Run all tests and provide detailed output.
    This is the main function to execute when running the script directly.
    """
    print("="*80)
    print("RUNNING DG MESH COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDGMesh))
    suite.addTests(loader.loadTestsFromTestCase(TestMeshFactory))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed! DG mesh implementation is correct.")
    else:
        print("\n⚠️  Some tests failed. Review the implementation.")
    
    return result.wasSuccessful()


def demonstrate_mesh_usage():
    """
    Demonstrate practical usage of the DG mesh classes.
    This function shows how to create meshes, validate them, and access DG data.
    """
    print("\n" + "="*80)
    print("DG MESH USAGE DEMONSTRATION")
    print("="*80)
    
    # Example 1: Create and validate a triangular mesh
    print("\n1. Triangular Cross Mesh:")
    tri_mesh = MeshFactory.create_triangular_cross_mesh()
    
    print(f"   - Vertices: {len(tri_mesh.vertices)}")
    print(f"   - Elements: {len(tri_mesh.cells)}")
    print(f"   - Boundary facets: {len(tri_mesh.boundary_facets)}")
    
    # Validate the mesh
    validator = MeshConnectivityValidator(tri_mesh)
    results = validator.run_all_tests()
    validator.print_results()
    
    # Example 2: Access DG-specific data
    print("\n2. DG-Specific Data Access:")
    print(f"   - Facet normals shape: {tri_mesh.facet_normals.shape}")
    print(f"   - Facet areas range: [{tri_mesh.facet_areas.min():.4f}, {tri_mesh.facet_areas.max():.4f}]")
    print(f"   - Sample left-right mapping (first 3 facets):")
    for i in range(min(3, len(tri_mesh.left_right_cells))):
        print(f"     Facet {i}: left={tri_mesh.left_right_cells[i, 0]}, right={tri_mesh.left_right_cells[i, 1]}")
    
    # Example 3: Create tetrahedral mesh
    print("\n3. Tetrahedral Mesh:")
    tet_mesh = MeshFactory.create_two_tetrahedra()
    
    print(f"   - Vertices: {len(tet_mesh.vertices)}")
    print(f"   - Elements: {len(tet_mesh.cells)}")
    print(f"   - Boundary facets: {len(tet_mesh.boundary_facets)}")
    print(f"   - Interior facets: {len(tet_mesh.facet_to_element) - len(tet_mesh.boundary_facets)}")
    
    # Example 4: Convert to JAX
    print("\n4. JAX Conversion:")
    jax_data = tri_mesh.to_jax_arrays()
    print(f"   - Converted {len(jax_data)} arrays to JAX")
    print(f"   - Sample JAX array types:")
    for key, value in list(jax_data.items())[:3]:
        print(f"     {key}: {value.shape}, dtype={value.dtype}")
    
    # Example 5: Mark custom boundary conditions
    print("\n5. Boundary Condition Marking:")
    boundary_facets = tri_mesh.boundary_facets
    if len(boundary_facets) >= 2:
        # Mark first two boundary facets as Neumann
        tri_mesh.mark_boundary_condition(boundary_facets[:2], 2)
        print(f"   - Marked facets {boundary_facets[:2]} as Neumann (type 2)")
        print(f"   - Other boundaries remain Dirichlet (type 1)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run tests
    success = run_comprehensive_tests()
    
    # Run demonstration
    demonstrate_mesh_usage()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)