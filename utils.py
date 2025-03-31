import numpy as np
import pdb
from constants import *

def mesh_reader_gambit_2d(file_name):
    # Purpose : Read in basic grid information to build grid
    # NOTE : gambit(Fluent, Inc) *.neu format is assumed
    with open(file_name, 'r') as fid:
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
    
    return Nv, VX, VY, K, EToV

def MeshReaderGambitBC2D(FileName):
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
    with open(FileName, 'r') as Fid:
        # read intro
        for i in range(6):
            line = Fid.readline()
        
        # Find number of nodes and number of elements
        dims = list(map(int, Fid.readline().split()))
        Nv = dims[0]
        K = dims[1]
        
        for i in range(2):
            line = Fid.readline()
        
        # read node coordinates
        VX = [0] * Nv
        VY = [0] * Nv
        
        for i in range(Nv):
            line = Fid.readline()
            tmpx = list(map(float, line.split()))
            VX[i] = tmpx[1]
            VY[i] = tmpx[2]
        
        for i in range(2):
            line = Fid.readline()
        
        # read element to node connectivity
        EToV = [[0, 0, 0] for _ in range(K)]
        
        for k in range(K):
            line = Fid.readline()
            tmpcon = list(map(float, line.split()))
            EToV[k] = [int(tmpcon[3]) - 1, int(tmpcon[4]) - 1, int(tmpcon[5]) - 1]
        
        # skip through material property section
        for i in range(4):
            line = Fid.readline()
        
        while "ENDOFSECTION" not in line:
            line = Fid.readline()
        
        line = Fid.readline()
        line = Fid.readline()
        
        # boundary codes
        BCType = [[0, 0, 0] for _ in range(K)]
        
        # Read all the boundary conditions at the nodes
        while line:
            if "In" in line:
                bcflag = In
            if "Out" in line:
                bcflag = Out
            if "Wall" in line:
                bcflag = Wall
            if "Far" in line:
                bcflag = Far
            if "Cyl" in line:
                bcflag = Cyl
            if "Dirichlet" in line:
                bcflag = Dirichlet
            if "Neuman" in line:
                bcflag = Neuman
            if "Slip" in line:
                bcflag = Slip
            
            line = Fid.readline()
            
            while line and "ENDOFSECTION" not in line:
                tmpid = list(map(int, line.split()))
                BCType[tmpid[0]-1][tmpid[2]-1] = bcflag  # Adjust for 0-indexing in Python
                line = Fid.readline()
            
            line = Fid.readline()
            if not line:
                break
            line = Fid.readline()
    
    return Nv, VX, VY, K, EToV, BCType