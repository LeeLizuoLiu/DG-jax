import numpy as np
import pdb
from constants import *
import scipy.io

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

def MeshReaderGambitBC2D(file_name):
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
            
            line = fid.readline()
            
            while line and "ENDOFSECTION" not in line:
                tmpid = list(map(int, line.split()))
                BCType[tmpid[0]-1,tmpid[2]-1] = bcflag  # Adjust for 0-indexing in Python
                line = fid.readline()
            
            line = fid.readline()
            if not line:
                break
            line = fid.readline()
    
    return Nv, VX, VY, K, EToV, BCType


def compare_matlab_numpy_results(matlab_file_path, elements_obj):
    """
    Compare variables between a MATLAB .mat file and NumPy computed results stored in an Elements class.
    
    Parameters:
    -----------
    matlab_file_path : str
        Path to the MATLAB .mat file containing the reference values
    elements_obj : Elements
        Instance of the Elements class containing the NumPy computed values
        
    Returns:
    --------
    dict
        Dictionary with comparison results for each variable
    """
    import scipy.io as sio
    import numpy as np
    
    # Load the MATLAB file
    mat_data = sio.loadmat(matlab_file_path)
    
    # Dictionary to store comparison results
    comparison_results = {}
    
    # List of variables to compare
    variables = [ 'mapM',
        'N', 'Np', 'Nfp', 'K', 'Nfaces', 'NODETOL', 'x', 'y', 'r', 's', 'V', 'invV', 
        'Dr', 'Ds', 'rx', 'sx', 'ry', 'sy', 'J', 'nx', 'ny', 'sJ', 'Fscale', 'EToE', 
        'EToF',  'mapP', 'vmapM', 'vmapP', 'vmapB', 'mapB', 'Vr', 'Vs', 'Drw', 
        'Dsw', 'LIFT', 'Nv', 'Fx', 'Fy'
    ]
    
    # Scalar variables (simple equality check)
    scalar_vars = ['N', 'Np', 'Nfp', 'K', 'Nfaces', 'NODETOL', 'Nv']
    # index variables
    index_vars = ['EToE', 'EToF', 'mapM', 'mapP', 'vmapM', 'vmapP', 'vmapB', 'mapB']
    # Array variables (need allclose check)
    array_vars = [var for var in variables if var not in scalar_vars]
    
    # Compare each variable
    for var_name in variables:
        if var_name not in mat_data:
            comparison_results[var_name] = {
                'status': 'missing',
                'message': f"Variable '{var_name}' not found in MATLAB file"
            }
            continue
            
        if not hasattr(elements_obj, var_name):
            comparison_results[var_name] = {
                'status': 'missing',
                'message': f"Variable '{var_name}' not found in Elements object"
            }
            continue
        
        matlab_val = mat_data[var_name]
        numpy_val = getattr(elements_obj, var_name)
        
        # Check if numpy_val is None
        if numpy_val is None:
            comparison_results[var_name] = {
                'status': 'missing',
                'message': f"Variable '{var_name}' is None in Elements object"
            }
            continue
        # index variables need to -1
        if var_name in index_vars:
            matlab_val = matlab_val - 1    
        # Handle special case for MATLAB column vectors vs NumPy arrays
        if len(np.shape(matlab_val)) == 2 and np.shape(matlab_val)[1] == 1:
            matlab_val = matlab_val.flatten(order="F")
        if len(np.shape(matlab_val)) == 1 and np.shape(matlab_val)[0] == 1:
            matlab_val = matlab_val[0]      
        # Check if shapes match
        if np.shape(matlab_val) != np.shape(numpy_val):
            comparison_results[var_name] = {
                'status': 'shape_mismatch',
                'message': f"Shape mismatch: MATLAB={np.shape(matlab_val)}, NumPy={np.shape(numpy_val)}"
            }
            continue
        
        # For scalar variables
        if var_name in scalar_vars:
            # Handle scalar values that might be in arrays
            matlab_scalar = matlab_val.item() if hasattr(matlab_val, 'item') else matlab_val
            numpy_scalar = numpy_val.item() if hasattr(numpy_val, 'item') else numpy_val
            
            if matlab_scalar == numpy_scalar:
                comparison_results[var_name] = {
                    'status': 'match',
                    'message': f"Exact match: {matlab_scalar}"
                }
            else:
                comparison_results[var_name] = {
                    'status': 'mismatch',
                    'message': f"Values differ: MATLAB={matlab_scalar}, NumPy={numpy_scalar}"
                }
        
        # For array variables
        else:
            try:
                # Convert to numpy arrays if they aren't already
                matlab_arr = np.array(matlab_val, dtype=np.float64)
                numpy_arr = np.array(numpy_val, dtype=np.float64)
                is_close = np.allclose(matlab_arr, numpy_arr, rtol=1e-13, atol=1e-10)
                max_diff = np.max(np.abs(matlab_arr - numpy_arr))
                max_index = np.argmax(np.abs(matlab_arr - numpy_arr))
                
                if is_close:
                    comparison_results[var_name] = {
                        'status': 'match',
                        'message': f"Arrays match within tolerance. Max difference: {max_diff}"
                    }
                else:
                    comparison_results[var_name] = {
                        'status': 'mismatch',
                        'message': f"Arrays differ. Max difference: {max_diff} at {max_index}"
                    }
            except Exception as e:
                comparison_results[var_name] = {
                    'status': 'error',
                    'message': f"Error comparing arrays: {str(e)}"
                }
    
    # Print summary
    print("\nComparison Summary:")
    print("===================")
    
    matches = sum(1 for var in comparison_results if comparison_results[var]['status'] == 'match')
    mismatches = sum(1 for var in comparison_results if comparison_results[var]['status'] == 'mismatch')
    missing = sum(1 for var in comparison_results if comparison_results[var]['status'] == 'missing')
    errors = sum(1 for var in comparison_results if comparison_results[var]['status'] in ['shape_mismatch', 'error'])
    
    print(f"Total variables: {len(variables)}")
    print(f"Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    print(f"Missing: {missing}")
    print(f"Errors: {errors}")
    
    print("\nDetailed Results:")
    print("================")
    for var_name in variables:
        if var_name in comparison_results:
            result = comparison_results[var_name]
            print(f"{var_name}: {result['status']} - {result['message']}")
    
    return comparison_results

# Example usage:
# elements = Elements(...)  # Your Elements class instance
# compare_matlab_numpy_results('startup.mat', elements)