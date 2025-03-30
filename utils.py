import numpy as np
import pdb

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