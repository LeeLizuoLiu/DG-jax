import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Union
from .LGLNodes import GaussLobatto1D 

# Type alias for JAX arrays
Array = jax.Array

@dataclass(frozen=True)
class QuadMesh:
    """
    Tensor-product Quadrilateral/Hexahedral Mesh.
    
    Data Layout for d=2:
      (Nx, Ny, P, P, ...)
    
    This mesh assumes a logically Cartesian topology (Structured Grid),
    even if the geometry is curvilinear (deformed).
    """
    
    # --- Geometry (Physical Coordinates)
    # Shape: [N1, ..., Nd, P, ..., P] (one array per dimension)
    coords: Tuple[Array, ...]
    
    # --- Geometric Factors (Metrics)
    # Jacobian of the mapping from reference to physical
    # Shape: [N1, ..., Nd, P, ..., P]
    J: Array  
    
    # Contravariant metric tensor (dxi/dx, dxi/dy, etc.)
    # Shape: [N1, ..., Nd, P, ..., P, d, d]
    # metrics[..., i, j] corresponds to d(xi_i) / d(x_j)
    metrics: Array

    # --- Reference Element Operators (1D)
    # These are broadcasted during computation to save memory
    r: Array      # [P] Nodes (LGL)
    w: Array      # [P] Quadrature weights
    D: Array      # [P, P] Differentiation matrix
    
    # --- Metadata
    ndim: int
    order: int    # Polynomial degree (N)
    P: int        # Number of nodes (N+1)
    
    # --- Grid Topology
    # Tuple of number of elements in each dimension: (Nx, Ny, ...)
    elements_shape: Tuple[int, ...]
    
    @property
    def n_elements(self) -> int:
        return int(np.prod(self.elements_shape))

    @property
    def mass_matrix_diagonal(self) -> Array:
        """
        Returns the diagonal of the global mass matrix.
        For LGL nodes, the mass matrix is effectively diagonal.
        
        M_global = J * (w_x \otimes w_y)
        """
        # 1. Compute tensor product of weights
        # We start with weights [P]
        # We need shape [P, P] (for 2D) or [P, P, P] (for 3D)
        
        weights_mesh = self.w
        for _ in range(self.ndim - 1):
            # Outer product to expand weights dimensionality
            # This effectively does w[i] * w[j] * ...
            weights_mesh = jnp.kron(weights_mesh, self.w)
            
        # Reshape to (P, P, ...)
        shape_p = (self.P,) * self.ndim
        weights_mesh = weights_mesh.reshape(shape_p)
        
        # 2. Broadcast weights to grid shape (Nx, Ny, ...)
        # J shape: (Nx, Ny, P, P)
        # weights shape needs to broadcast against J
        
        # We rely on JAX broadcasting. 
        # J: [Nx, Ny, P, P]
        # weights: [P, P] -> broadcasts to [1, 1, P, P]
        
        return self.J * weights_mesh

    def integrate_over_domain(self, u: Array) -> Array:
        """
        Integrates a variable u over the entire domain.
        Integral = sum(u * mass_matrix_diagonal)
        
        Args:
            u: Shape [Nx, Ny, ..., P, P, ..., n_vars]
        """
        # Expand mass matrix to have a singleton dim for n_vars
        mass = self.mass_matrix_diagonal[..., jnp.newaxis]
        
        # Weighted sum
        return jnp.sum(u * mass, axis=tuple(range(u.ndim - 1)))

    @classmethod
    def from_domain_box(
        cls,
        bounds: List[Tuple[float, float]], # [(x_min, x_max), (y_min, y_max), ...]
        n_elems: List[int],                # [Nx, Ny, ...]
        order: int
    ) -> "QuadMesh":
        """
        Factory: Creates a rectilinear mesh on a box domain.
        Adaptive for 1D, 2D, or 3D based on lengths of inputs.
        """
        ndim = len(bounds)
        assert len(n_elems) == ndim
        
        # 1. Generate Reference 1D LGL nodes
        r, D, w = GaussLobatto1D(order)
        P = order + 1
        
        # 2. Generate Grid Coordinates
        # We generate linear spaces for element boundaries
        # Then map LGL nodes into each element
        
        mesh_coords = []
        
        # We need to build the full (Nx, Ny, P, P) coordinate arrays
        # Strategy: Build 1D arrays for each dim, then meshgrid them
        
        coords_1d_list = []
        
        for d in range(ndim):
            x_min, x_max = bounds[d]
            ne = n_elems[d]
            
            # Element boundaries
            e_bounds = np.linspace(x_min, x_max, ne + 1)
            
            # Size of each element
            dx = (x_max - x_min) / ne
            
            # Map r [-1, 1] to physical element [x_i, x_{i+1}]
            # x = center + r * (dx/2)
            centers = (e_bounds[:-1] + e_bounds[1:]) / 2.0
            
            # Shape: [ne, 1] + [1, P] -> [ne, P]
            nodes_1d = centers[:, None] + r[None, :] * (dx / 2.0)
            coords_1d_list.append(nodes_1d)

        # 3. Create Tensor Product Grid
        # We use meshgrid with 'indexing=ij'
        # coords_1d_list[0] is [Nx, P]
        # coords_1d_list[1] is [Ny, P]
        
        # We need to construct the full Nd arrays.
        # This requires a bit of reshaping magic to interleave N and P dimensions
        # Example 2D: We want (Nx, Ny, Px, Py).
        # np.meshgrid gives us (Nx, P, Ny, P) roughly, we need to transpose.
        
        # Let's generalize.
        # Create grids for element indices (Nx, Ny) and Node indices (P, P)
        
        # Helper to broadcast 1D coordinate arrays to full shape
        full_coords = []
        for d in range(ndim):
            # Start with shape [N_d, P]
            c = coords_1d_list[d] 
            
            # We want final shape [N1, N2... Nd, P, P... P]
            # Reshape c to align with the d-th N-dimension and d-th P-dimension
            
            # Construct reshape pattern
            shape = [1] * (2 * ndim)
            shape[d] = n_elems[d]          # The element dim
            shape[ndim + d] = P            # The node dim
            
            c_reshaped = c.reshape(shape)
            
            # Now tile/broadcast to full shape
            tile_reps = list(n_elems) + [P] * ndim
            tile_reps[d] = 1
            tile_reps[ndim + d] = 1
            
            full_c = jnp.tile(c_reshaped, tile_reps)
            full_coords.append(full_c)
            
        coords = tuple(full_coords)

        # 4. Compute Metrics (Jacobian and dxi/dx)
        # For a rectilinear box, this is diagonal and constant per dimension
        # J = Product(dx/2)
        # dxi/dx = 2/dx
        
        # Compute dx for each dim
        deltas = np.array([(b[1]-b[0])/n for b, n in zip(bounds, n_elems)])
        
        # Geometric Jacobian J (determinant)
        # Value is product(dx/2)
        detJ_val = np.prod(deltas / 2.0)
        J_array = jnp.full(full_coords[0].shape, detJ_val)
        
        # Metric Tensor (Inverse Jacobian matrix dxi/dx)
        # Shape [..., d, d]
        # For box: Diagonal matrix with entries 2/dx
        metrics_shape = full_coords[0].shape + (ndim, ndim)
        metrics_array = jnp.zeros(metrics_shape)
        
        for d in range(ndim):
            val = 2.0 / deltas[d]
            # Set diagonal d,d
            metrics_array = metrics_array.at[..., d, d].set(val)

        return cls(
            coords=coords,
            J=J_array,
            metrics=metrics_array,
            r=jnp.array(r),
            w=jnp.array(w),
            D=jnp.array(D),
            ndim=ndim,
            order=order,
            P=P,
            elements_shape=tuple(n_elems)
        )
    
