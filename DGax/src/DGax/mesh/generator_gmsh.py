import gmsh
import numpy as np
from typing import Tuple, Dict
# Add this new import at the top
import matplotlib.pyplot as plt
import matplotlib.tri as tri

"""
The gmsh script for testing periodic mesh generation
For more general usage, A general mesh generator is on the way.
"""


def generate_periodic_square_mesh(
    h: float = 0.1,
    domain_size: float = 1.0,
    output_file: str = None,
    visualize: bool = False 
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate a periodic square mesh [0,1]x[0,1] using Gmsh.
    
    Args:
        h: Characteristic mesh size
        domain_size: Side length of square domain
        output_file: Optional .msh file to save (for visualization)
    
    Returns:
        VX: Vertex x-coordinates [Nv]
        VY: Vertex y-coordinates [Nv]
        EToV: Element-to-vertex connectivity [K, 3] (0-indexed)
        periodic_node_map: Dict mapping slave nodes to master nodes
    """
    gmsh.initialize()
    gmsh.model.add("periodic_square")
    gmsh.option.setNumber("General.Terminal", 1)
    
    # ================================================================
    # 1. Create geometry
    # ================================================================
    L = domain_size
    p1 = gmsh.model.geo.addPoint(0, 0, 0, h)
    p2 = gmsh.model.geo.addPoint(L, 0, 0, h)
    p3 = gmsh.model.geo.addPoint(L, L, 0, h)
    p4 = gmsh.model.geo.addPoint(0, L, 0, h)
    
    # Lines (keep tags for BC assignment)
    bottom = gmsh.model.geo.addLine(p1, p2)  # y=0
    right  = gmsh.model.geo.addLine(p2, p3)  # x=1
    top    = gmsh.model.geo.addLine(p4, p3)  # y=1
    left   = gmsh.model.geo.addLine(p1, p4)  # x=0
    
    # Surface
    loop = gmsh.model.geo.addCurveLoop([bottom, right, -top, -left])
    surface = gmsh.model.geo.addPlaneSurface([loop])
    
    gmsh.model.geo.synchronize()
    
    # ================================================================
    # 2. Set periodicity using affine transformations
    # ================================================================
    # Left (master) -> Right (slave): translation by (L, 0, 0)
    # Affine matrix in row-major form: [R | t; 0 0 0 1]
    translation_x = [
        1, 0, 0, L,   # [1, 0, 0] * x + L
        0, 1, 0, 0,   # [0, 1, 0] * y + 0
        0, 0, 1, 0,   # [0, 0, 1] * z + 0
        0, 0, 0, 1    # Homogeneous coordinate
    ]
    
    # Bottom (master) -> Top (slave): translation by (0, L, 0)
    translation_y = [
        1, 0, 0, 0,
        0, 1, 0, L,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]
    
    gmsh.model.mesh.setPeriodic(1, [right], [left], translation_x)
    gmsh.model.mesh.setPeriodic(1, [top], [bottom], translation_y)
    
    print(f"✓ Periodicity set: left({left})↔right({right}), bottom({bottom})↔top({top})")
    
    # ================================================================
    # 3. Mesh generation
    # ================================================================
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(2)
    
    if output_file:
        gmsh.write(output_file)
        print(f"✓ Mesh written to {output_file}")

    # ================================================================
    # 4. Extract mesh data
    # ================================================================
    # Get all nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)
    
    # Create node tag -> index mapping (Gmsh uses 1-indexed tags)
    node_tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
    Nv = len(node_tags)
    VX = node_coords[:, 0]
    VY = node_coords[:, 1]
    
    # Get triangular elements (type 2 in Gmsh)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surface)
    
    # Find triangle element type
    tri_type_idx = list(elem_types).index(2)
    tri_node_tags = elem_node_tags[tri_type_idx].reshape(-1, 3)
    # Convert to 0-indexed
    EToV = np.array([[node_tag_to_idx[tag] for tag in elem] 
                     for elem in tri_node_tags], dtype=int)
    
    K = EToV.shape[0]
    print(f"✓ Extracted mesh: {Nv} nodes, {K} triangles")

    # ================================================================
    # Build edge dictionary from EToV for global edge indices
    # ================================================================
    edge_dict = {}
    edge_counter = 0
    
    for tri_idx, triangle in enumerate(EToV):
        v0, v1, v2 = triangle
        edges = [
            (min(v0, v1), max(v0, v1)),  # Edge 0
            (min(v1, v2), max(v1, v2)),  # Edge 1  
            (min(v2, v0), max(v2, v0))   # Edge 2
        ]
        
        for edge in edges:
            if edge not in edge_dict:
                edge_dict[edge] = edge_counter
                edge_counter += 1
    
    # Reverse mapping: global edge index to edge tuple
    global_edge_to_tuple = {v: k for k, v in edge_dict.items()}
    
    print(f"✓ Built edge dictionary: {len(edge_dict)} unique edges")
    
    # ================================================================
    # 5. Extract periodic node correspondence
    # ================================================================
    periodic_edge_map = []
    gmsh_to_global_edge = {}
    master_edges_list = []
    slave_edges_list = []
    master_edge_nodes_list = []
    slave_edge_nodes_list = []
    
    # Get periodic correspondence for edges (dimension 1)
    # try:
    # For each master edge, get corresponding slave nodes
    for master_edge in [left, bottom]:
        slave_edge = right if master_edge == left else top

        # Get periodic correspondence
        # slave_master_map = gmsh.model.mesh.getPeriodicNodes(1, slave_edge)
        master_tag, slave_nodes, master_nodes, transf = gmsh.model.mesh.getPeriodicNodes(1, slave_edge)
        slave_tag, _, _, _ = gmsh.model.mesh.getPeriodicNodes(1, master_edge)

        # Get 1D elements (edges) on master and slave
        elem_types_m, master_edges, master_edge_nodes = gmsh.model.mesh.getElements(1, master_edge)
        elem_types_s, slave_edges,  slave_edge_nodes  = gmsh.model.mesh.getElements(1, slave_edge)

        master_edges = np.array(master_edges[0])
        slave_edges = np.array(slave_edges[0])
        master_edge_nodes = np.array(master_edge_nodes[0]).reshape(-1, 2)
        slave_edge_nodes = np.array(slave_edge_nodes[0]).reshape(-1, 2)

        # Store for visualization and verification
        master_edges_list.extend(master_edges)
        slave_edges_list.extend(slave_edges)
        master_edge_nodes_list.extend(master_edge_nodes)
        slave_edge_nodes_list.extend(slave_edge_nodes)

        for i, (gmsh_edge_tag, nodes) in enumerate(zip(master_edges, master_edge_nodes)):
            # Convert Gmsh node tags to vertex indices
            v0_idx = node_tag_to_idx[nodes[0]]
            v1_idx = node_tag_to_idx[nodes[1]]
            
            # Create sorted edge tuple (for dictionary lookup)
            edge_tuple = tuple(sorted([v0_idx, v1_idx]))
            
            # Get global edge index
            if edge_tuple in edge_dict:
                global_edge_idx = edge_dict[edge_tuple]
                gmsh_to_global_edge[gmsh_edge_tag] = global_edge_idx
            else:
                print(f"Warning: Edge {edge_tuple} not found in edge_dict")

        for i, (gmsh_edge_tag, nodes) in enumerate(zip(slave_edges, slave_edge_nodes)):
            # Convert Gmsh node tags to vertex indices
            v0_idx = node_tag_to_idx[nodes[0]]
            v1_idx = node_tag_to_idx[nodes[1]]
            
            # Create sorted edge tuple (for dictionary lookup)
            edge_tuple = tuple(sorted([v0_idx, v1_idx]))
            
            # Get global edge index
            if edge_tuple in edge_dict:
                global_edge_idx = edge_dict[edge_tuple]
                gmsh_to_global_edge[gmsh_edge_tag] = global_edge_idx
            else:
                print(f"Warning: Edge {edge_tuple} not found in edge_dict")

        # Build periodic_edge_map using GLOBAL edge indices
        for master_gmsh_tag, slave_gmsh_tag in zip(master_edges, slave_edges):
            if master_gmsh_tag in gmsh_to_global_edge and slave_gmsh_tag in gmsh_to_global_edge:
                master_global = gmsh_to_global_edge[master_gmsh_tag]
                slave_global = gmsh_to_global_edge[slave_gmsh_tag]
                periodic_edge_map.append((master_global, slave_global))

        # For visual verification purposes
        if visualize:
            print(periodic_edge_map)
            visualize_periodic_mapping(master_edge_nodes, slave_edge_nodes, 
                                       node_coords, tri_node_tags, 
                                       master_edges, slave_edges, 
                                       master_nodes, slave_nodes, 
                                       master_tag, slave_tag,
                                       'periodic_mapping_left.png' if master_edge == left else 'periodic_mapping_top.png')

    print(f"✓ Found {len(periodic_edge_map)} periodic edge pairs")
    # except Exception as e:
    #     print(f"Warning: Could not extract periodic edges: {e}")
    #     periodic_edge_map = []
    
    # ================================================================
    # 6. Create boundary condition maps using physical coordinates
    # ================================================================
    # Since Gmsh handles periodicity internally, we need to identify
    # boundary nodes by their coordinates
    bc_node_map = None

    if visualize:
        # Visualize periodic edge mapping with global indices
        visualize_periodic_edge_mapping_global(
            VX, VY, EToV, periodic_edge_map, global_edge_to_tuple,
            filename='periodic_global_edges.png'
        )        
    gmsh.finalize()
    
    return VX, VY, EToV, periodic_edge_map, global_edge_to_tuple, bc_node_map    

# Add this function after verify_edge_correspondence in generator_gmsh.py:
def visualize_periodic_edge_mapping_global(VX, VY, EToV, periodic_edge_map, 
                                          global_edge_to_tuple, filename='periodic_global_edges.png'):
    """
    Visualize periodic edge mapping using global edge indices.
    
    Args:
        VX, VY: Vertex coordinates
        EToV: Element-to-vertex connectivity
        periodic_edge_map: Dictionary mapping global edge indices
        global_edge_to_tuple: Mapping from global edge index to (v0, v1)
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create triangulation
    triang = tri.Triangulation(VX, VY, triangles=EToV)
    
    # Plot mesh
    ax.triplot(triang, 'k-', linewidth=0.5, alpha=0.3)
    
    # Plot periodic edge pairs
    colors = plt.cm.tab10(np.linspace(0, 1, len(periodic_edge_map)//2))
    color_idx = 0
    
    # We need to avoid double counting (since map is bidirectional)
    plotted_pairs = set()
    
    for edge1, edge2 in periodic_edge_map:
        # Avoid double counting
        if (edge2, edge1) in plotted_pairs:
            continue
            
        # Get edge tuples
        if edge1 in global_edge_to_tuple and edge2 in global_edge_to_tuple:
            edge1_tuple = global_edge_to_tuple[edge1]
            edge2_tuple = global_edge_to_tuple[edge2]
            
            # Calculate midpoints
            mid1_x = (VX[edge1_tuple[0]] + VX[edge1_tuple[1]]) / 2
            mid1_y = (VY[edge1_tuple[0]] + VY[edge1_tuple[1]]) / 2
            mid2_x = (VX[edge2_tuple[0]] + VX[edge2_tuple[1]]) / 2
            mid2_y = (VY[edge2_tuple[0]] + VY[edge2_tuple[1]]) / 2
            
            # Plot edge midpoints
            color = colors[color_idx % len(colors)]
            ax.plot(mid1_x, mid1_y, 'o', markersize=8, color=color, alpha=0.7)
            ax.plot(mid2_x, mid2_y, 's', markersize=8, color=color, alpha=0.7)
            
            # Label with global edge indices
            ax.text(mid1_x, mid1_y, f'E{edge1}', 
                   fontsize=9, fontweight='bold', color='white',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9))
            ax.text(mid2_x, mid2_y, f'E{edge2}', 
                   fontsize=9, fontweight='bold', color='white',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9))
            
            # Draw connecting line
            ax.plot([mid1_x, mid2_x], [mid1_y, mid2_y], 
                   '--', color=color, linewidth=2, alpha=0.5)
            
            plotted_pairs.add((edge1, edge2))
            color_idx += 1
    
    # Format plot
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Periodic Edge Mapping with Global Indices\n{len(plotted_pairs)} edge pairs')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ Periodic edge mapping (global indices) saved to {filename}")

def visualize_periodic_mapping(master_edge_nodes, slave_edge_nodes, 
                               node_coords, tri_node_tags, 
                               master_edges, slave_edges, 
                               master_nodes, slave_nodes, 
                               master_tag, slave_tag,
                               filename):

    import matplotlib.pyplot as plt
    # ------------------------------------------------------------
    # Plot mesh
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.triplot(node_coords[:, 0], node_coords[:, 1], tri_node_tags - 1, 'k-', lw=0.5, alpha=0.3)

    # ------------------------------------------------------------
    # 1. Label ALL mesh edges (optional)
    # ------------------------------------------------------------
    def get_edge_midpoint(edge_nodes):
        """Calculate midpoint of an edge given node indices"""
        return node_coords[edge_nodes - 1].mean(axis=0)

    # Uncomment to label *all* 1D elements (can be cluttered)
    """
    all_edges = np.unique(np.concatenate([master_edges, slave_edges]))
    for edge_tag in all_edges:
        # Find which curve this edge belongs to
        if edge_tag in master_edges:
            nodes_idx = master_edge_nodes[np.where(master_edges == edge_tag)[0][0]]
            color = 'blue'
        else:
            nodes_idx = slave_edge_nodes[np.where(slave_edges == edge_tag)[0][0]]
            color = 'red'

        midpoint = get_edge_midpoint(nodes_idx)
        plt.text(midpoint[0], midpoint[1], f'{edge_tag}', 
                 color=color, fontsize=6, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    """

    # ------------------------------------------------------------
    # 2. Label ONLY periodic boundary edges
    # ------------------------------------------------------------
    def label_periodic_edges(edge_tags, edge_nodes, color, label_prefix):
        """Label edges on a specific boundary"""
        for edge_tag, node_pair in zip(edge_tags, edge_nodes):
            midpoint = get_edge_midpoint(node_pair)
            plt.text(midpoint[0], midpoint[1], f'{label_prefix}{edge_tag}', 
                     color=color, fontsize=8, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', 
                              facecolor=color, alpha=0.2))

    # Label master edges (blue)
    label_periodic_edges(master_edges, master_edge_nodes, 'blue', 'M')

    # Label slave edges (red)
    label_periodic_edges(slave_edges, slave_edge_nodes, 'red', 'S')

    # ------------------------------------------------------------
    # 3. Draw verification connectors between corresponding edge midpoints
    # ------------------------------------------------------------
    for master_e, slave_e in zip(master_edges, slave_edges):
        # Get midpoints
        master_mid = get_edge_midpoint(master_edge_nodes[np.where(master_edges == master_e)[0][0]])
        slave_mid = get_edge_midpoint(slave_edge_nodes[np.where(slave_edges == slave_e)[0][0]])

        # Draw dashed line connecting corresponding edges
        plt.plot([master_mid[0], slave_mid[0]], 
                 [master_mid[1], slave_mid[1]], 
                 'g--', lw=1, alpha=0.5)

    # ------------------------------------------------------------
    # 4. Highlight boundary nodes for reference
    # ------------------------------------------------------------
    plt.plot(tri_node_tags[master_nodes - 1, 0], tri_node_tags[master_nodes - 1, 1], 
             'bo', markersize=6, label=f'Master nodes (curve {master_tag})')
    plt.plot(tri_node_tags[slave_nodes - 1, 0], tri_node_tags[slave_nodes - 1, 1], 
             'ro', markersize=6, label=f'Slave nodes (curve {slave_tag})')

    # Legend & formatting
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.title(f'Periodic Edge Mapping: Master(Blue) ↔ Slave(Red)\nGreen dashed lines show correspondence')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(filename)
    plt.close()



# Add these new visualization functions after visualize_periodic_mapping function

def visualize_mesh_with_triangle_indices(VX, VY, EToV, filename='triangle_indices.png'):
    """
    Visualize the mesh with triangle indices labeled.
    
    Args:
        VX, VY: Vertex coordinates
        EToV: Element-to-vertex connectivity
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create triangulation
    triang = tri.Triangulation(VX, VY, triangles=EToV)
    
    # Plot mesh
    ax.triplot(triang, 'k-', linewidth=0.5, alpha=0.5)
    ax.plot(VX, VY, 'o', markersize=3, color='blue', alpha=0.5)
    
    # Calculate triangle centers for labels
    for i, triangle in enumerate(EToV):
        # Get vertices of triangle i
        v0, v1, v2 = triangle
        # Calculate centroid
        centroid_x = (VX[v0] + VX[v1] + VX[v2]) / 3.0
        centroid_y = (VY[v0] + VY[v1] + VY[v2]) / 3.0
        
        # Add triangle index label
        ax.text(centroid_x, centroid_y, str(i), 
                fontsize=8, fontweight='bold', color='red',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', 
                         facecolor='yellow', alpha=0.7))
    
    # Format plot
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Mesh with Triangle Indices (Total: {len(EToV)} triangles)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ Triangle indices visualization saved to {filename}")

def visualize_mesh_with_face_indices(VX, VY, EToV, filename='face_indices.png'):
    """
    Visualize the mesh with face (edge) indices labeled.
    
    Args:
        VX, VY: Vertex coordinates
        EToV: Element-to-vertex connectivity
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create triangulation
    triang = tri.Triangulation(VX, VY, triangles=EToV)
    
    # Plot mesh
    ax.triplot(triang, 'k-', linewidth=0.5, alpha=0.5)
    ax.plot(VX, VY, 'o', markersize=3, color='blue', alpha=0.5)
    
    # Build a dictionary of unique faces (edges) with their indices
    face_dict = {}
    face_counter = 0
    
    for tri_idx, triangle in enumerate(EToV):
        # Get vertices
        v0, v1, v2 = triangle
        
        # Define faces (edges) of triangle
        faces = [
            (min(v0, v1), max(v0, v1)),  # Face 0
            (min(v1, v2), max(v1, v2)),  # Face 1  
            (min(v2, v0), max(v2, v0))   # Face 2
        ]
        
        # Add faces to dictionary and assign unique indices
        for local_face_idx, face in enumerate(faces):
            if face not in face_dict:
                face_dict[face] = {
                    'global_idx': face_counter,
                    'triangles': [(tri_idx, local_face_idx)]
                }
                face_counter += 1
            else:
                face_dict[face]['triangles'].append((tri_idx, local_face_idx))
    
    # Plot face indices
    for face, data in face_dict.items():
        v1_idx, v2_idx = face
        # Calculate midpoint of the face
        mid_x = (VX[v1_idx] + VX[v2_idx]) / 2.0
        mid_y = (VY[v1_idx] + VY[v2_idx]) / 2.0
        
        # Check if it's a boundary face (only belongs to one triangle)
        is_boundary = len(data['triangles']) == 1
        color = 'red' if is_boundary else 'blue'
        
        # Add face index label
        ax.text(mid_x, mid_y, str(data['global_idx']), 
                fontsize=7, fontweight='bold', color=color,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', 
                         facecolor='white', alpha=0.8))
    
    # Format plot
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Mesh with Face Indices (Total: {face_counter} faces, Blue: interior, Red: boundary)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ Face indices visualization saved to {filename}")
    
    return face_dict

# In generator_gmsh.py, replace the visualize_mesh_full function with this enhanced version:
def visualize_mesh_full(VX, VY, EToV, master_edges=None, master_edge_nodes=None, 
                       slave_edges=None, slave_edge_nodes=None, 
                       node_tag_to_idx=None, tri_node_tags=None,
                       filename='mesh_full.png'):
    """
    Visualize mesh with triangle, vertex, and edge indices.
    
    Args:
        VX, VY: Vertex coordinates
        EToV: Element-to-vertex connectivity
        master_edges: List of master edge tags from Gmsh (optional)
        master_edge_nodes: Node connectivity for master edges (optional)
        slave_edges: List of slave edge tags from Gmsh (optional)
        slave_edge_nodes: Node connectivity for slave edges (optional)
        node_tag_to_idx: Mapping from Gmsh node tags to indices (optional)
        tri_node_tags: Gmsh triangle node tags (optional)
        filename: Output filename
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Create triangulation
    triang = tri.Triangulation(VX, VY, triangles=EToV)
    
    # -------------------------------------------------------------
    # Left subplot: Triangle indices
    # -------------------------------------------------------------
    ax1.triplot(triang, 'k-', linewidth=0.5, alpha=0.5)
    
    # Label triangle indices
    for i, triangle in enumerate(EToV):
        centroid_x = (VX[triangle[0]] + VX[triangle[1]] + VX[triangle[2]]) / 3.0
        centroid_y = (VY[triangle[0]] + VY[triangle[1]] + VY[triangle[2]]) / 3.0
        ax1.text(centroid_x, centroid_y, f'T{i}', 
                fontsize=7, fontweight='bold', color='red',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    ax1.set_aspect('equal')
    ax1.set_title(f'Triangle Indices (Total: {len(EToV)})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------
    # Middle subplot: Vertex indices
    # -------------------------------------------------------------
    ax2.triplot(triang, 'k-', linewidth=0.5, alpha=0.5)
    ax2.plot(VX, VY, 'o', markersize=4, color='blue', alpha=0.7)
    
    # Label vertex indices
    for i, (x, y) in enumerate(zip(VX, VY)):
        ax2.text(x, y, str(i), 
                fontsize=8, fontweight='bold', color='darkgreen',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax2.set_aspect('equal')
    ax2.set_title(f'Vertex Indices (Total: {len(VX)})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------
    # Right subplot: Edge indices with Gmsh tags
    # -------------------------------------------------------------
    ax3.triplot(triang, 'k-', linewidth=0.5, alpha=0.3)
    
    # First, compute all edges from EToV and assign global edge indices
    edge_dict = {}
    edge_counter = 0
    
    for tri_idx, triangle in enumerate(EToV):
        # Get vertices
        v0, v1, v2 = triangle
        
        # Define faces (edges) of triangle
        edges = [
            (min(v0, v1), max(v0, v1)),  # Edge 0
            (min(v1, v2), max(v1, v2)),  # Edge 1  
            (min(v2, v0), max(v2, v0))   # Edge 2
        ]
        
        # Add edges to dictionary and assign unique indices
        for local_edge_idx, edge in enumerate(edges):
            if edge not in edge_dict:
                edge_dict[edge] = {
                    'global_idx': edge_counter,
                    'triangles': [(tri_idx, local_edge_idx)],
                    'midpoint': ((VX[edge[0]] + VX[edge[1]]) / 2, 
                                 (VY[edge[0]] + VY[edge[1]]) / 2)
                }
                edge_counter += 1
            else:
                edge_dict[edge]['triangles'].append((tri_idx, local_edge_idx))
    
    # Plot edge indices (global from EToV)
    for edge, data in edge_dict.items():
        mid_x, mid_y = data['midpoint']
        # Check if it's a boundary edge (only belongs to one triangle)
        is_boundary = len(data['triangles']) == 1
        color = 'red' if is_boundary else 'blue'
        
        # Add global edge index label
        ax3.text(mid_x, mid_y, f'E{data["global_idx"]}', 
                fontsize=7, fontweight='bold', color=color,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', 
                         facecolor='white', alpha=0.8))
    
    # Overlay Gmsh edge tags if available
    if (master_edges is not None and master_edge_nodes is not None and 
        slave_edges is not None and slave_edge_nodes is not None and
        node_tag_to_idx is not None):
        
        # Helper function to convert Gmsh node tags to vertex indices
        def gmsh_nodes_to_vertex_indices(gmsh_nodes):
            """Convert Gmsh node tags to vertex indices"""
            return [node_tag_to_idx[tag] for tag in gmsh_nodes]
        
        # Process master edges
        for i, (edge_tag, nodes) in enumerate(zip(master_edges, master_edge_nodes)):
            # Convert Gmsh nodes to vertex indices
            v0_idx = node_tag_to_idx[nodes[0]]
            v1_idx = node_tag_to_idx[nodes[1]]
            
            # Create sorted edge tuple (for dictionary lookup)
            edge_tuple = tuple(sorted([v0_idx, v1_idx]))
            
            # Get midpoint
            mid_x = (VX[v0_idx] + VX[v1_idx]) / 2
            mid_y = (VY[v0_idx] + VY[v1_idx]) / 2
            
            # Check if this edge is in our edge_dict
            if edge_tuple in edge_dict:
                global_idx = edge_dict[edge_tuple]['global_idx']
                # Draw a green box around the corresponding global edge index
                rect = plt.Rectangle((mid_x-0.015, mid_y-0.015), 0.03, 0.03,
                                    linewidth=2, edgecolor='green', 
                                    facecolor='none', alpha=0.8)
                ax3.add_patch(rect)
                
                # Add Gmsh tag label
                ax3.text(mid_x, mid_y+0.02, f'M{edge_tag}→E{global_idx}', 
                        fontsize=6, fontweight='bold', color='darkgreen',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.1', 
                                 facecolor='lightgreen', alpha=0.8))
        
        # Process slave edges
        for i, (edge_tag, nodes) in enumerate(zip(slave_edges, slave_edge_nodes)):
            # Convert Gmsh nodes to vertex indices
            v0_idx = node_tag_to_idx[nodes[0]]
            v1_idx = node_tag_to_idx[nodes[1]]
            
            # Create sorted edge tuple (for dictionary lookup)
            edge_tuple = tuple(sorted([v0_idx, v1_idx]))
            
            # Get midpoint
            mid_x = (VX[v0_idx] + VX[v1_idx]) / 2
            mid_y = (VY[v0_idx] + VY[v1_idx]) / 2
            
            # Check if this edge is in our edge_dict
            if edge_tuple in edge_dict:
                global_idx = edge_dict[edge_tuple]['global_idx']
                # Draw an orange box around the corresponding global edge index
                rect = plt.Rectangle((mid_x-0.015, mid_y-0.015), 0.03, 0.03,
                                    linewidth=2, edgecolor='orange', 
                                    facecolor='none', alpha=0.8)
                ax3.add_patch(rect)
                
                # Add Gmsh tag label
                ax3.text(mid_x, mid_y-0.02, f'S{edge_tag}→E{global_idx}', 
                        fontsize=6, fontweight='bold', color='darkorange',
                        ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.1', 
                                 facecolor='wheat', alpha=0.8))
        
        ax3.set_title(f'Edge Indices (Total: {edge_counter})\nGreen: Master edges, Orange: Slave edges')
    else:
        ax3.set_title(f'Edge Indices (Total: {edge_counter}, Red: boundary, Blue: interior)')
    
    ax3.set_aspect('equal')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Full mesh visualization saved to {filename}")

# Add this function after visualize_mesh_full in generator_gmsh.py:
def verify_edge_correspondence(VX, VY, EToV, master_edges, master_edge_nodes, 
                              slave_edges, slave_edge_nodes, node_tag_to_idx):
    """
    Verify that Gmsh edge tags correspond correctly to global edge indices from EToV.
    
    Returns:
        Dict with verification results
    """
    # Build global edge dictionary from EToV
    edge_dict = {}
    edge_counter = 0
    
    for tri_idx, triangle in enumerate(EToV):
        v0, v1, v2 = triangle
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0))
        ]
        
        for edge in edges:
            if edge not in edge_dict:
                edge_dict[edge] = {
                    'global_idx': edge_counter,
                    'triangles': []
                }
                edge_counter += 1
            edge_dict[edge]['triangles'].append(tri_idx)
    
    results = {
        'master_matches': [],
        'slave_matches': [],
        'master_missing': [],
        'slave_missing': []
    }
    
    # Helper function to find edge in dictionary
    def find_edge(nodes):
        v0_idx = node_tag_to_idx[nodes[0]]
        v1_idx = node_tag_to_idx[nodes[1]]
        edge_tuple = tuple(sorted([v0_idx, v1_idx]))
        return edge_tuple, edge_dict.get(edge_tuple)
    
    # Check master edges
    print("\n" + "="*60)
    print("Verifying Master Edge Correspondence:")
    print("="*60)
    for i, (edge_tag, nodes) in enumerate(zip(master_edges, master_edge_nodes)):
        edge_tuple, edge_data = find_edge(nodes)
        if edge_data:
            results['master_matches'].append((edge_tag, edge_data['global_idx']))
            print(f"  Master edge M{edge_tag} → Global edge E{edge_data['global_idx']}")
        else:
            results['master_missing'].append(edge_tag)
            print(f"  ✗ Master edge M{edge_tag} not found in EToV edges")
    
    # Check slave edges
    print("\n" + "="*60)
    print("Verifying Slave Edge Correspondence:")
    print("="*60)
    for i, (edge_tag, nodes) in enumerate(zip(slave_edges, slave_edge_nodes)):
        edge_tuple, edge_data = find_edge(nodes)
        if edge_data:
            results['slave_matches'].append((edge_tag, edge_data['global_idx']))
            print(f"  Slave edge S{edge_tag} → Global edge E{edge_data['global_idx']}")
        else:
            results['slave_missing'].append(edge_tag)
            print(f"  ✗ Slave edge S{edge_tag} not found in EToV edges")
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Total edges from EToV: {len(edge_dict)}")
    print(f"Master edges matched: {len(results['master_matches'])}/{len(master_edges)}")
    print(f"Slave edges matched: {len(results['slave_matches'])}/{len(slave_edges)}")
    
    if results['master_missing']:
        print(f"Missing master edges: {results['master_missing']}")
    if results['slave_missing']:
        print(f"Missing slave edges: {results['slave_missing']}")
    
def create_neu_format_with_periodic(
    VX: np.ndarray,
    VY: np.ndarray, 
    EToV: np.ndarray,
    bc_node_map: Dict[str, np.ndarray],
    output_file: str = "periodic_square.neu"
):
    """
    Convert mesh to Gambit .neu format with periodic BC tags.
    
    BC codes:
        0: Interior
        9: Periodic (left boundary, master)
        10: Periodic (right boundary, slave)
        11: Periodic (bottom boundary, master)
        12: Periodic (top boundary, slave)
    """
    Nv = len(VX)
    K = EToV.shape[0]
    
    # Create BCType array: [K, 3] with BC code for each face
    BCType = np.zeros((K, 3), dtype=int)
    
    # Build edge-to-BC mapping
    edge_bc = {}
    tol = 1e-10
    
    for k in range(K):
        for f in range(3):
            # Get edge vertices
            v1 = EToV[k, f]
            v2 = EToV[k, (f+1) % 3]
            
            # Sort for consistent edge identification
            edge = tuple(sorted([v1, v2]))
            
            # Check if edge is on boundary
            x1, y1 = VX[v1], VY[v1]
            x2, y2 = VX[v2], VY[v2]
            
            bc_code = 0  # Interior by default
            
            # Check each boundary
            if abs(x1) < tol and abs(x2) < tol:
                bc_code = 9  # Left (master)
            elif abs(x1 - 1.0) < tol and abs(x2 - 1.0) < tol:
                bc_code = 10  # Right (slave)
            elif abs(y1) < tol and abs(y2) < tol:
                bc_code = 11  # Bottom (master)
            elif abs(y1 - 1.0) < tol and abs(y2 - 1.0) < tol:
                bc_code = 12  # Top (slave)
            
            BCType[k, f] = bc_code
            if bc_code > 0:
                edge_bc[edge] = bc_code
    
    # Write .neu file
    with open(output_file, 'w') as f:
        # Header
        f.write("        CONTROL INFO 2.0.0\n")
        f.write("** GAMBIT NEUTRAL FILE\n")
        f.write("Periodic Square Mesh\n")
        f.write("PROGRAM:                Gmsh     VERSION:  4.11\n")
        f.write("\n")
        f.write("   NUMNP   NELEM   NGRPS  NBSETS   NDFCD   NDFVL\n")
        f.write(f"{Nv:8d}{K:8d}       0       0       2       2\n")
        f.write("ENDOFSECTION\n")
        
        # Nodal coordinates
        f.write("   NODAL COORDINATES 2.0.0\n")
        for i in range(Nv):
            f.write(f"{i+1:10d}  {VX[i]:20.11e}  {VY[i]:20.11e}\n")
        f.write("ENDOFSECTION\n")
        
        # Elements
        f.write("      ELEMENTS/CELLS 2.0.0\n")
        for k in range(K):
            f.write(f"{k+1:8d}  2  3  {EToV[k,0]+1:8d}{EToV[k,1]+1:8d}{EToV[k,2]+1:8d}\n")
        f.write("ENDOFSECTION\n")
        
        # Boundary conditions
        f.write("       BOUNDARY CONDITIONS 1.0.0\n")
        
        # Count BC faces
        n_bc_faces = np.sum(BCType != 0)
        
        if n_bc_faces > 0:
            bc_names = {9: "PeriodicLeft", 10: "PeriodicRight", 
                       11: "PeriodicBottom", 12: "PeriodicTop"}
            
            for bc_code, bc_name in bc_names.items():
                faces_with_bc = np.argwhere(BCType == bc_code)
                if len(faces_with_bc) > 0:
                    f.write(f"{bc_name} {bc_code} {len(faces_with_bc)} 0 0\n")
                    for elem_idx, face_idx in faces_with_bc:
                        f.write(f"{elem_idx+1:8d}       3       {face_idx+1}\n")
        
        f.write("ENDOFSECTION\n")
    
    print(f"✓ Wrote .neu file: {output_file}")
    return BCType


if __name__ == "__main__":
    # Generate mesh
    VX, VY, EToV, periodic_map, bc_nodes, global_edge_to_tuple = generate_periodic_square_mesh(
        h=0.1,
        domain_size=1.0,
        # output_file="periodic_square.msh",
        visualize=True
    )
    
    print("\n" + "="*60)
    print("Mesh generation complete!")
    print(f"Vertices: {len(VX)}")
    print(f"Elements: {EToV.shape[0]}")
    print(f"Periodic pairs: {len(periodic_map)}")
    print("="*60)