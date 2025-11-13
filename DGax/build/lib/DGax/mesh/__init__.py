# src/Dgax/mesh/__init__.py
"""Mesh module for DGAX"""
from .geometry import MeshGeometry2D, compute_physical_coordinates
from .connectivity import MeshConnectivity2D, connect_elements_2d
from .io import read_gambit_neu

__all__ = [
    "MeshGeometry2D",
    "MeshConnectivity2D", 
    "read_gambit_neu",
    "compute_physical_coordinates",
    "connect_elements_2d"
]