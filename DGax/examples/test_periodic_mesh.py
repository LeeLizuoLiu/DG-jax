# examples/setup_mesh.py
from DGax.mesh import TriMesh 
import numpy as np
import pdb

mesh = TriMesh.from_gmsh_periodic(
    order=4,
    h=0.1,
    domain_size=1.0
)


