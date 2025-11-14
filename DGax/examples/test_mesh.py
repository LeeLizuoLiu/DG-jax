# examples/setup_mesh.py
from DGax.mesh import TriMesh 

# 1. read file 
mesh_path = "Mesh_neu/test.neu"
order = 2

# 3. construct connectivity 
import time
start = time.time()
mesh = TriMesh.from_gambit(mesh_path, order)
end = time.time()
# print(f"Mesh construction time: {end - start:.4f} seconds")
# print(mesh.bc_maps)