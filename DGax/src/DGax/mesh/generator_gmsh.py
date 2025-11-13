import gmsh
import numpy as np
from .TriMesh import TriMesh

# This is the code to generate a periodic square with triangle mesh.

def generate_periodic_square_mesh():
    # ===================================================================
    # 1. Initialize & create geometry (explicit tagging)
    # ===================================================================
    gmsh.initialize()
    gmsh.model.add("periodic_square")

    # Create points
    p1 = gmsh.model.occ.addPoint(0, 0, 0)
    p2 = gmsh.model.occ.addPoint(1, 0, 0)
    p3 = gmsh.model.occ.addPoint(1, 1, 0)
    p4 = gmsh.model.occ.addPoint(0, 1, 0)

    # Create lines (we keep their tags)
    bottom = gmsh.model.occ.addLine(p1, p2)   # tag = bottom
    right  = gmsh.model.occ.addLine(p2, p3)   # tag = right
    top    = gmsh.model.occ.addLine(p4, p3)   # tag = top
    left   = gmsh.model.occ.addLine(p1, p4)   # tag = left

    # Create surface
    loop = gmsh.model.occ.addCurveLoop([bottom, right, top, left])
    square = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()

    print(f"✓ Geometry created: left={left}, right={right}, bottom={bottom}, top={top}")

    # ===================================================================
    # 2. Set periodicity with 4×4 transformation matrices
    # ===================================================================
    # Left (master) → Right (slave):  translation by (1,0,0)
    # Matrix = [R | t; 0 0 0 1] flattened row-major
    # R = I (identity 3×3), t = (1,0,0)

    gmsh.model.mesh.setPeriodic(
        1, 
        [right],           # slave
        [left],            # master
        [1,0,0,1,          # row 0: [1,0,0, 1]
         0,1,0,0,          # row 1: [0,1,0, 0]
         0,0,1,0,          # row 2: [0,0,1, 0]
         0,0,0,1]          # row 3: [0,0,0, 1]
    )

    # Bottom (master) → Top (slave):  translation by (0,1,0)
    gmsh.model.mesh.setPeriodic(
        1,
        [top],             # slave
        [bottom],          # master
        [1,0,0,0,          # row 0: [1,0,0, 0]
         0,1,0,1,          # row 1: [0,1,0, 1]
         0,0,1,0,          # row 2: [0,0,1, 0]
         0,0,0,1]          # row 3: [0,0,0, 1]
    )

    print("✓ Periodicity constraints applied")

    # ===================================================================
    # 3. Mesh settings & generation
    # ===================================================================
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.08)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.08)
    gmsh.option.setNumber("Mesh.Algorithm", 6)          # Frontal-Delaunay
    gmsh.model.mesh.generate(2)

    # ===================================================================
    # 4. Extract mesh directly into NumPy arrays (NO file I/O)
    # ===================================================================
    _, xyz, _ = gmsh.model.mesh.getNodes()
    nodes = xyz.reshape(-1, 3)                          # (nNodes, 3)

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements()
    tri_idx = list(elem_types).index(2)                 # locate triangle block
    triangles = elem_node_tags[tri_idx].reshape(-1, 3) - 1  # 0‑based

    print(f"✓ Mesh extracted: {nodes.shape[0]} nodes, {triangles.shape[0]} triangles")

    # ===================================================================
    # 5. Obtain periodicity mapping
    # ===================================================================
    # Get 1D elements (edges) on master and slave
    elem_types_m, bottom_edges, master_edge_nodes = gmsh.model.mesh.getElements(1, bottom)
    elem_types_s, top_edges, slave_edge_nodes = gmsh.model.mesh.getElements(1, top)
    elem_types_m, left_edges, master_edge_nodes = gmsh.model.mesh.getElements(1, left)
    elem_types_s, right_edges, slave_edge_nodes = gmsh.model.mesh.getElements(1, right)   

    # Elements correspond by index: elem_tags_m[i] <-> elem_tags_s[i]
    for i, (master_elem, slave_elem) in enumerate(zip(bottom_edges[0], top_edges[0])):
        print(f"Bottom edge {master_elem} ↔ Top edge {slave_elem}")
    for i, (master_elem, slave_elem) in enumerate(zip(left_edges[0], right_edges[0])):
        print(f"Left edge {master_elem} ↔ Right edge {slave_elem}")
       
    # ===================================================================
    # 6. Clean up (kernel released, nothing written to disk)
    # ===================================================================
    gmsh.finalize()

    # ===================================================================
    # 7. Use the mesh
    # ===================================================================
    # Your solver code here: nodes and triangles are ready