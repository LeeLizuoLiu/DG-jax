import abc
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Set
import unittest
import sys

class DGMesh(abc.ABC):
    """
    Abstract base class for Discontinuous Galerkin (DG) method meshes.
    
    This class provides a common interface and shared implementation for various
    mesh types (triangular, tetrahedral, etc.) used in DG methods. It stores
    all necessary connectivity information and geometric data required for 
    DG flux calculations and element interactions.
    
    In the Mesh file, the measure(area/length, volume) of both the element and each facets should be evaluated
    """
    
    def __init__(self, vertices: np.ndarray, cells: np.ndarray):
        """
        Initialize base mesh data.
        
        Args:
            vertices: Array of vertex coordinates, shape (num_vertices, dim)
            cells: Array of element connectivity, shape (num_cells, vertices_per_cell)
        """
        self.vertices = vertices  # Vertex coordinates
        self.cells = cells        # Element-to-vertex connectivity
        self.dim = vertices.shape[1]  # Spatial dimension
        
        # Core connectivity arrays (initialized as None)
        self.element_to_element: Optional[np.ndarray] = None  # Element neighbor connectivity
        self.element_to_facet: Optional[np.ndarray] = None    # Element-to-facet mapping
        self.facet_to_element: Optional[np.ndarray] = None    # Facet-to-element mapping
        self.boundary_facets: Optional[List[int]] = None      # List of boundary facet indices
        
        # DG-specific data
        self.facet_normals: Optional[np.ndarray] = None       # Facet normal vectors
        self.facet_areas: Optional[np.ndarray] = None         # Facet areas/lengths
        self.facet_centroids: Optional[np.ndarray] = None     # Facet centroids
        self.left_right_cells: Optional[np.ndarray] = None    # Left-right cell indices for each facet
        self.facet_boundary_types: Optional[np.ndarray] = None # Boundary condition types
        
        # Build all connectivity and DG data
        self.build_connectivity()
    
    @abc.abstractmethod
    def get_facets(self, cell: np.ndarray) -> List[Tuple[int, ...]]:
        """
        Get all facets (edges in 2D, faces in 3D) for a given cell.
        
        Args:
            cell: Array of vertex indices for the cell
            
        Returns:
            List of facets, where each facet is a tuple of vertex indices
        """
        pass
    
    @abc.abstractmethod
    def get_facet_type(self) -> str:
        """Return facet type: 'edge' for 2D, 'face' for 3D."""
        pass
    
    @abc.abstractmethod
    def num_facets_per_element(self) -> int:
        """Return number of facets per element (e.g., 3 for triangle, 4 for tetrahedron)."""
        pass
    
    @abc.abstractmethod
    def _compute_facet_geometry_impl(self, facet_vertices: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Compute geometry for a single facet.
        
        Args:
            facet_vertices: Array of vertex coordinates for the facet
            
        Returns:
            Tuple of (normal_vector, area, centroid)
        """
        pass
    
    def build_connectivity(self):
        """Build complete connectivity information including DG-specific data."""
        # Build core element connectivity (implemented by subclasses)
        self._build_element_connectivity()
        
        # Build DG-specific data (common implementation)
        self._compute_facet_geometry()
        self._build_left_right_mapping()
        self._initialize_boundary_types()
    
    @abc.abstractmethod
    def _build_element_connectivity(self):
        """Build element-to-element and element-to-facet connectivity."""
        pass
    
    def _compute_facet_geometry(self):
        """
        Compute geometric properties for all facets (normals, areas, centroids).
        This method works for any dimension and calls the abstract implementation
        for actual geometry calculations.
        """
        num_facets = len(self.facet_to_element)
        self.facet_normals = np.zeros((num_facets, self.dim))
        self.facet_areas = np.zeros(num_facets)
        self.facet_centroids = np.zeros((num_facets, self.dim))
        
        for facet_idx in range(num_facets):
            # Get the first cell that uses this facet
            cell_idx = self.facet_to_element[facet_idx, 0]
            if cell_idx == -1:
                continue
            
            # Get vertex coordinates for this facet
            facet_vertices_coords = self._get_facet_vertices_coords(facet_idx)
            
            # Compute geometry using subclass implementation
            normal, area, centroid = self._compute_facet_geometry_impl(facet_vertices_coords)
            
            self.facet_normals[facet_idx] = normal
            self.facet_areas[facet_idx] = area
            self.facet_centroids[facet_idx] = centroid
    
    def _get_facet_vertices_coords(self, facet_idx: int) -> np.ndarray:
        """
        Get vertex coordinates for a facet.
        
        Args:
            facet_idx: Global facet index
            
        Returns:
            Array of vertex coordinates for the facet
        """
        # Find which cell and local facet index corresponds to this global facet
        cell_idx = self.facet_to_element[facet_idx, 0]
        local_facet_indices = np.where(self.element_to_facet[cell_idx] == facet_idx)[0]
        
        if len(local_facet_indices) == 0:
            raise ValueError(f"Facet {facet_idx} not found in element {cell_idx}")
        
        local_facet_idx = local_facet_indices[0]
        cell_vertices = self.cells[cell_idx]
        facets = self.get_facets(cell_vertices)
        facet_vertex_indices = facets[local_facet_idx]
        
        return self.vertices[list(facet_vertex_indices)]
    
    def _build_left_right_mapping(self):
        """
        Build left-right cell mapping for each facet.
        
        For interior facets: left cell has smaller index, right cell has larger index.
        For boundary facets: left cell is the interior cell, right cell is -1.
        This convention is crucial for consistent flux computations in DG methods.
        """
        num_facets = len(self.facet_to_element)
        self.left_right_cells = -np.ones((num_facets, 2), dtype=int)
        
        for facet_idx in range(num_facets):
            cells = self.facet_to_element[facet_idx]
            
            if cells[1] == -1:  # Boundary facet
                self.left_right_cells[facet_idx, 0] = cells[0]
                # Right cell remains -1
            else:  # Interior facet
                # Ensure consistent ordering: smaller index on left
                left_cell, right_cell = sorted(cells)
                self.left_right_cells[facet_idx, 0] = left_cell
                self.left_right_cells[facet_idx, 1] = right_cell
    
    def _initialize_boundary_types(self):
        """
        Initialize boundary condition types.
        
        Types:
            0: Interior facet
            1: Dirichlet boundary
            2: Neumann boundary
            3: Periodic boundary
        """
        num_facets = len(self.facet_to_element)
        self.facet_boundary_types = np.zeros(num_facets, dtype=int)
        
        # Mark boundary facets as Dirichlet by default
        self.facet_boundary_types[self.boundary_facets] = 1
    
    def mark_boundary_condition(self, facet_indices: List[int], bc_type: int):
        """
        Manually mark boundary facets with specific condition types.
        
        Args:
            facet_indices: List of facet indices to mark
            bc_type: Boundary condition type (1=Dirichlet, 2=Neumann, 3=Periodic)
        """
        if self.facet_boundary_types is None:
            self._initialize_boundary_types()
        
        # Validate that marked facets are actually boundary facets
        invalid_facets = set(facet_indices) - set(self.boundary_facets)
        if invalid_facets:
            raise ValueError(f"Facets {invalid_facets} are not boundary facets")
        
        self.facet_boundary_types[list(facet_indices)] = bc_type
    
    def to_jax_arrays(self) -> Dict[str, jnp.ndarray]:
        """
        Convert all mesh data to JAX arrays for GPU acceleration.
        
        Returns:
            Dictionary of JAX arrays containing all mesh data
        """
        jax_data = {}
        
        # Basic mesh data
        if self.vertices is not None:
            jax_data['vertices'] = jnp.array(self.vertices)
        if self.cells is not None:
            jax_data['cells'] = jnp.array(self.cells)
        if self.element_to_element is not None:
            jax_data['element_to_element'] = jnp.array(self.element_to_element)
        if self.element_to_facet is not None:
            jax_data['element_to_facet'] = jnp.array(self.element_to_facet)
        if self.facet_to_element is not None:
            jax_data['facet_to_element'] = jnp.array(self.facet_to_element)
        
        # DG-specific data
        if self.facet_normals is not None:
            jax_data['facet_normals'] = jnp.array(self.facet_normals)
        if self.facet_areas is not None:
            jax_data['facet_areas'] = jnp.array(self.facet_areas)
        if self.facet_centroids is not None:
            jax_data['facet_centroids'] = jnp.array(self.facet_centroids)
        if self.left_right_cells is not None:
            jax_data['left_right_cells'] = jnp.array(self.left_right_cells)
        if self.facet_boundary_types is not None:
            jax_data['facet_boundary_types'] = jnp.array(self.facet_boundary_types)
        
        return jax_data
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive mesh statistics.
        
        Returns:
            Dictionary containing mesh statistics
        """
        stats = {
            'num_vertices': len(self.vertices),
            'num_cells': len(self.cells),
            'num_facets': len(self.facet_to_element) if self.facet_to_element is not None else 0,
            'num_boundary_facets': len(self.boundary_facets) if self.boundary_facets is not None else 0,
            'dimension': self.dim,
            'facet_type': self.get_facet_type(),
            'elements_per_cell': self.cells.shape[1]
        }
        
        if self.element_to_element is not None:
            neighbor_counts = np.sum(self.element_to_element != -1, axis=1)
            stats['avg_neighbors_per_cell'] = float(np.mean(neighbor_counts))
            stats['min_neighbors'] = int(np.min(neighbor_counts))
            stats['max_neighbors'] = int(np.max(neighbor_counts))
        
        if self.facet_areas is not None:
            stats['total_boundary_area'] = float(np.sum(self.facet_areas[self.boundary_facets]))
            stats['min_facet_area'] = float(np.min(self.facet_areas))
            stats['max_facet_area'] = float(np.max(self.facet_areas))
        
        return stats


class TriangularMesh(DGMesh):
    """Concrete implementation for 2D triangular meshes."""
    
    def get_facets(self, cell: np.ndarray) -> List[Tuple[int, int]]:
        """Return the three edges of a triangle."""
        return [
            (int(cell[0]), int(cell[1])),
            (int(cell[1]), int(cell[2])),
            (int(cell[2]), int(cell[0]))
        ]
    
    def get_facet_type(self) -> str:
        return "edge"
    
    def num_facets_per_element(self) -> int:
        return 3
    
    def _compute_facet_geometry_impl(self, facet_vertices: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Compute geometry for a 2D edge.
        
        Args:
            facet_vertices: Array of shape (2, 2) containing coordinates of edge endpoints
            
        Returns:
            Tuple of (normal_vector, length, centroid)
        """
        v1, v2 = facet_vertices
        edge_vec = v2 - v1
        length = np.linalg.norm(edge_vec)
        
        # Compute normal vector (pointing outward by convention)
        normal = np.array([-edge_vec[1], edge_vec[0]])
        if length > 1e-12:
            normal /= length
        
        centroid = (v1 + v2) / 2.0
        
        return normal, length, centroid
    
    def _build_element_connectivity(self):
        """
        Build connectivity for triangular mesh.
        
        This method constructs:
        - element_to_element: neighbor connectivity
        - element_to_facet: element-to-facet mapping
        - facet_to_element: facet-to-element mapping
        - boundary_facets: list of boundary facet indices
        """
        num_cells = len(self.cells)
        facets_per_cell = self.num_facets_per_element()
        
        self.element_to_vertex = self.cells.copy()
        # Initialize arrays
        self.element_to_element = -np.ones((num_cells, facets_per_cell), dtype=int)
        self.element_to_facet = -np.ones((num_cells, facets_per_cell), dtype=int)
        
        # Dictionary to track facets: sorted vertex tuple -> (cell_idx, local_facet_idx)
        facet_dict: Dict[Tuple[int, int], Tuple[int, int]] = {}
        facet_counter = 0
        facet_to_element_data = []
        
        # First pass: identify all facets and neighbor relationships
        for cell_idx, cell in enumerate(self.cells):
            facets = self.get_facets(cell)
            
            for local_facet_idx, facet in enumerate(facets):
                sorted_facet = tuple(sorted(facet))
                
                if sorted_facet in facet_dict:
                    # Facet already exists - this is an interior facet
                    neighbor_cell, neighbor_facet_idx = facet_dict[sorted_facet]
                    
                    # Set up neighbor connectivity
                    self.element_to_element[cell_idx, local_facet_idx] = neighbor_cell
                    self.element_to_element[neighbor_cell, neighbor_facet_idx] = cell_idx
                    
                    # Share the same facet index
                    shared_facet_idx = self.element_to_facet[neighbor_cell, neighbor_facet_idx]
                    self.element_to_facet[cell_idx, local_facet_idx] = shared_facet_idx
                    
                    # Update facet-to-element mapping for the neighbor
                    facet_to_element_data[shared_facet_idx]['cells'].append(cell_idx)
                    
                else:
                    # New facet - add to dictionary and assign index
                    facet_dict[sorted_facet] = (cell_idx, local_facet_idx)
                    facet_idx = facet_counter
                    facet_counter += 1
                    
                    self.element_to_facet[cell_idx, local_facet_idx] = facet_idx
                    
                    facet_to_element_data.append({
                        'vertices': sorted_facet,
                        'cells': [cell_idx]
                    })
        
        # Build facet_to_element array
        num_facets = len(facet_to_element_data)
        self.facet_to_element = -np.ones((num_facets, 2), dtype=int)
        
        for facet_idx, data in enumerate(facet_to_element_data):
            cells = data['cells']
            for i, cell_idx in enumerate(cells):
                if i < 2:  # At most 2 cells share a facet
                    self.facet_to_element[facet_idx, i] = cell_idx
        
        # Identify boundary facets (those with only one adjacent cell)
        self.boundary_facets = []
        for facet_idx in range(num_facets):
            if self.facet_to_element[facet_idx, 1] == -1:
                self.boundary_facets.append(facet_idx)


class TetrahedralMesh(DGMesh):
    """Concrete implementation for 3D tetrahedral meshes."""
    
    def get_facets(self, cell: np.ndarray) -> List[Tuple[int, int, int]]:
        """Return the four triangular faces of a tetrahedron."""
        return [
            (int(cell[0]), int(cell[1]), int(cell[2])),
            (int(cell[0]), int(cell[1]), int(cell[3])),
            (int(cell[0]), int(cell[2]), int(cell[3])),
            (int(cell[1]), int(cell[2]), int(cell[3]))
        ]
    
    def get_facet_type(self) -> str:
        return "face"
    
    def num_facets_per_element(self) -> int:
        return 4
    
    def _compute_facet_geometry_impl(self, facet_vertices: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Compute geometry for a 3D triangular face.
        
        Args:
            facet_vertices: Array of shape (3, 3) containing coordinates of triangle vertices
            
        Returns:
            Tuple of (normal_vector, area, centroid)
        """
        v1, v2, v3 = facet_vertices
        
        # Compute normal vector using cross product
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        
        # Compute area
        area = 0.5 * np.linalg.norm(normal)
        
        # Normalize normal vector
        if area > 1e-12:
            normal /= (2.0 * area)  # Normalization factor for area-weighted normal
        
        centroid = (v1 + v2 + v3) / 3.0
        
        return normal, area, centroid
    
    def _build_element_connectivity(self):
        """
        Build connectivity for tetrahedral mesh.
        
        Logic is analogous to triangular mesh but handles tetrahedral faces.
        """
        num_cells = len(self.cells)
        facets_per_cell = self.num_facets_per_element()
        
        self.element_to_vertex = self.cells.copy()
        # Initialize arrays
        self.element_to_element = -np.ones((num_cells, facets_per_cell), dtype=int)
        self.element_to_facet = -np.ones((num_cells, facets_per_cell), dtype=int)
        
        # Track facets using sorted vertex tuples
        facet_dict: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
        facet_counter = 0
        facet_to_element_data = []
        
        # Process all cells
        for cell_idx, cell in enumerate(self.cells):
            facets = self.get_facets(cell)
            
            for local_facet_idx, facet in enumerate(facets):
                sorted_facet = tuple(sorted(facet))
                
                if sorted_facet in facet_dict:
                    # Interior facet found
                    neighbor_cell, neighbor_facet_idx = facet_dict[sorted_facet]
                    
                    self.element_to_element[cell_idx, local_facet_idx] = neighbor_cell
                    self.element_to_element[neighbor_cell, neighbor_facet_idx] = cell_idx
                    
                    shared_facet_idx = self.element_to_facet[neighbor_cell, neighbor_facet_idx]
                    self.element_to_facet[cell_idx, local_facet_idx] = shared_facet_idx
                    
                    facet_to_element_data[shared_facet_idx]['cells'].append(cell_idx)
                    
                else:
                    # New facet
                    facet_dict[sorted_facet] = (cell_idx, local_facet_idx)
                    facet_idx = facet_counter
                    facet_counter += 1
                    
                    self.element_to_facet[cell_idx, local_facet_idx] = facet_idx
                    
                    facet_to_element_data.append({
                        'vertices': sorted_facet,
                        'cells': [cell_idx]
                    })
        
        # Build facet_to_element array
        num_facets = len(facet_to_element_data)
        self.facet_to_element = -np.ones((num_facets, 2), dtype=int)
        
        for facet_idx, data in enumerate(facet_to_element_data):
            cells = data['cells']
            for i, cell_idx in enumerate(cells):
                if i < 2:  # At most 2 cells share a facet
                    self.facet_to_element[facet_idx, i] = cell_idx
        
        # Identify boundary facets
        self.boundary_facets = []
        for facet_idx in range(num_facets):
            if self.facet_to_element[facet_idx, 1] == -1:
                self.boundary_facets.append(facet_idx)


class MeshFactory:
    """Factory class for creating various test meshes."""
    
    @staticmethod
    def create_triangular_cross_mesh() -> TriangularMesh:
        """
        Create a simple cross-shaped triangular mesh for testing.
        
        Returns:
            TriangularMesh with 5 vertices and 4 elements
        """
        vertices = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [1.0, 1.0], [0.5, 0.5]
        ], dtype=float)
        
        cells = np.array([
            [0, 1, 4],
            [1, 3, 4],
            [3, 2, 4],
            [2, 0, 4]
        ], dtype=int)
        
        return TriangularMesh(vertices, cells)
    
    @staticmethod
    def create_triangular_unit_square(nx: int = 2, ny: int = 2) -> TriangularMesh:
        """
        Create a structured triangular mesh of a unit square.
        
        Args:
            nx: Number of divisions in x-direction
            ny: Number of divisions in y-direction
            
        Returns:
            TriangularMesh of the unit square
        """
        from itertools import product
        
        # Generate vertices
        vertices = []
        for j in range(ny + 1):
            for i in range(nx + 1):
                vertices.append([i / nx, j / ny])
        vertices = np.array(vertices, dtype=float)
        
        # Generate cells (two triangles per rectangle)
        cells = []
        for j in range(ny):
            for i in range(nx):
                v0 = j * (nx + 1) + i
                v1 = v0 + 1
                v2 = v0 + nx + 1
                v3 = v2 + 1
                
                # Two triangles per rectangle
                cells.append([v0, v1, v2])
                cells.append([v1, v3, v2])
        
        cells = np.array(cells, dtype=int)
        return TriangularMesh(vertices, cells)
    
    @staticmethod
    def create_single_tetrahedron() -> TetrahedralMesh:
        """
        Create a single tetrahedron mesh.
        
        Returns:
            TetrahedralMesh with 4 vertices and 1 element
        """
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=float)
        
        cells = np.array([[0, 1, 2, 3]], dtype=int)
        
        return TetrahedralMesh(vertices, cells)
    
    @staticmethod
    def create_two_tetrahedra() -> TetrahedralMesh:
        """
        Create a mesh with two adjacent tetrahedra sharing a face.
        
        Returns:
            TetrahedralMesh with 5 vertices and 2 elements
        """
        vertices = np.array([
            [0.0, 0.0, 0.0],  # v0
            [1.0, 0.0, 0.0],  # v1
            [0.0, 1.0, 0.0],  # v2
            [0.0, 0.0, 1.0],  # v3
            [1.0, 1.0, 1.0]   # v4
        ], dtype=float)
        
        # Two tetrahedra sharing the face (v1, v2, v3)
        cells = np.array([
            [0, 1, 2, 3],  # First tetrahedron
            [1, 2, 3, 4]   # Second tetrahedron sharing face (1,2,3)
        ], dtype=int)
        
        return TetrahedralMesh(vertices, cells)

