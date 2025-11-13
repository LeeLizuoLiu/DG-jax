# src/dg_jax/mesh/io.py
"""Mesh file I/O utilities"""
import numpy as np
from typing import Tuple, Dict, Any
import re

def read_gambit_neu(mesh_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Read in basic grid information to build grid
    NOTE: gambit(Fluent, Inc) *.neu format is assumed
    
    Args:
        FileName: Path to the .neu file
        
    Returns:
        Nv: Number of vertices
        VX: x-coordinates of vertices
        VY: y-coordinates of vertices
        K: Number of elements
        EToV: Element to vertex connectivity
        BCType: Boundary condition types
    """
    # Purpose : Read in basic grid information to build grid
    # NOTE : gambit(Fluent, Inc) *.neu format is assumed
    with open(mesh_path, 'r') as fid:
        # Read intro (skip the first 6 lines)
        for _ in range(6):
            fid.readline()
        # Find number of nodes and number of elements
        dims = np.array(fid.readline().split(), dtype=int)
        Nv = dims[0]
        K = dims[1]
        
        # Skip next 2 lines
        for _ in range(2):
            fid.readline()
        
        # Read node coordinates
        VX = np.zeros(Nv)
        VY = np.zeros(Nv)
        for i in range(Nv):
            line = fid.readline()
            tmpx = np.array(line.split(), dtype=float)
            VX[i] = tmpx[1]
            VY[i] = tmpx[2]
        
        # Skip next 2 lines
        for _ in range(2):
            fid.readline()
        
        # Read element to node connectivity
        EToV = np.zeros((K, 3), dtype=int)
        for k in range(K):
            line = fid.readline()
            tmpcon = np.array(line.split(), dtype=float)
            EToV[k, 0] = int(tmpcon[3]) - 1
            EToV[k, 1] = int(tmpcon[4]) - 1
            EToV[k, 2] = int(tmpcon[5]) - 1
        
        # skip through material property section
        for i in range(4):
            line = fid.readline()
        
        while "ENDOFSECTION" not in line:
            line = fid.readline()
        
        line = fid.readline()
        line = fid.readline()
        
        # boundary codes
        BCType = np.zeros((K, 3), dtype=int)
        
        # Read all the boundary conditions at the nodes
        while line:
            bc_name = line.strip().strip('"')
            # Map BC name to integer code
            bc_code = _bc_name_to_code(bc_name)
            # if "In" in line:
            #     bcflag = In
            # if "Out" in line:
            #     bcflag = Out
            # if "Wall" in line:
            #     bcflag = Wall
            # if "Far" in line:
            #     bcflag = Far
            # if "Cyl" in line:
            #     bcflag = Cyl
            # if "Dirichlet" in line:
            #     bcflag = Dirichlet
            # if "Neuman" in line:
            #     bcflag = Neuman
            # if "Slip" in line:
            #     bcflag = Slip
            
            line = fid.readline()
            
            while line and "ENDOFSECTION" not in line:
                tmpid = list(map(int, line.split()))
                BCType[tmpid[0]-1,tmpid[2]-1] = bc_code  # Adjust for 0-indexing in Python
                line = fid.readline()
            
            line = fid.readline()
            if not line:
                break
            line = fid.readline()
    
    return VX, VY, EToV, BCType

    # Parse boundary sections
    # line_idx = bc_start
    # while line_idx < len(lines):
    #     line = lines[line_idx].strip()
    #     if "ENDOFSECTION" in line:
    #         break
    #     if re.match(r'^\s*\d+\s*$', line):  # BC section header
    #         n_bc_faces = int(line)
    #         line_idx += 1
    #         bc_name = lines[line_idx].strip().strip('"')
    #         line_idx += 1
    #         
    #         # Map BC name to integer code
    #         bc_code = _bc_name_to_code(bc_name)
    #         bc_regions[bc_name] = bc_code
    #         
    #         # Read faces
    #         for _ in range(n_bc_faces):
    #             line_idx += 1
    #             parts = lines[line_idx].strip().split()
    #             elem_id = int(parts[2]) - 1  # 0-indexed
    #             face_id = int(parts[3]) - 1  # 0-indexed
    #             BCType[face_id, elem_id] = bc_code
    #     line_idx += 1
    
    # return VX, VY, EToV, BCType

def _bc_name_to_code(name: str) -> int:
    """Map BC name to integer code"""
    BC_MAP = {
        "In": 1, "Out": 2, "Wall": 3, "Far": 4,
        "Cyl": 5, "Dirichlet": 6, "Neuman": 7, "Slip": 8
    }
    return BC_MAP.get(name, 0)  # 0 for interior