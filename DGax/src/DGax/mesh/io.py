# src/dg_jax/mesh/io.py
"""Mesh file I/O utilities"""
import numpy as np
from typing import Tuple, Dict, Any
import re
import pdb

"""
The read file should return the following:
        VX: x-coordinates of vertices
        VY: y-coordinates of vertices
        EToV: Element to vertex connectivity
        BCType: Boundary condition types
"""



def read_gambit_neu(mesh_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Read in basic grid information to build grid
    NOTE: gambit(Fluent, Inc) *.neu format is assumed
    
    Args:
        FileName: Path to the .neu file
        
    Returns:
        VX: x-coordinates of vertices
        VY: y-coordinates of vertices
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
        bc_code = 0  # Default to interior (0) until we find a BC definition
        while line:
            stripped = line.strip()

            # Exit when section ends
            if not stripped:
                break
            
            # Determine line type
            if stripped and not stripped[0].isdigit():
                # **BC DEFINITION LINE**: Extract new BC type
                # Example: "Wall 1 32 0 0" → bc_name = "Wall"
                bc_name = stripped.split()[0].strip('"')
                bc_code = _bc_name_to_code(bc_name)
            elif stripped:
                # **DATA LINE**: Parse with CURRENT bc_code
                # Example: "3    3    2"
                tmpid = list(map(int, line.split()))
                BCType[tmpid[0]-1, tmpid[2]-1] = bc_code

            line = fid.readline()

    return VX, VY, EToV, BCType

def _bc_name_to_code(name: str) -> int:
    """Map BC name to integer code"""
    BC_MAP = {
        "Inflow": 1, "Outflow": 2, "Wall": 3, "Far": 4,
        "Cyl": 5, "Dirichlet": 6, "Neuman": 7, "Slip": 8
    }
    return BC_MAP.get(name, 0)  # 0 for interior