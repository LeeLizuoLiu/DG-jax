# src/dg_jax/mesh/base.py
"""Abstract base for all mesh types"""
from typing import Protocol, runtime_checkable
from jax import Array

@runtime_checkable
class Mesh(Protocol):
    """Protocol that all mesh types must satisfy"""
    
    # --- Geometry attributes (common to all mesh types)
    x: Array
    J: Array
    face_normals: Array
    face_scale: Array
    
    # --- Connectivity attributes (common to all mesh types)
    vmapM: Array
    vmapP: Array
    mapB: Array
    bc_maps: dict[str, Array]
    
    # --- type hints for attributes that vary by mesh
    @property
    def ndim(self) -> int:
        """Spatial dimension"""
        ...
    
    @property
    def n_elements(self) -> int:
        """Number of elements"""
        ...
    
    @property
    def n_nodes_per_element(self) -> int:
        """Number of nodes per element (Np)"""
        ...
    
    def get_boundary_mask(self, bc_type: str) -> Array:
        """Get indices for specific boundary type"""
        ...