# DGax: Discontinuous Galerkin Solver for 2D Euler Equations

## Overview
This project implements a Discontinuous Galerkin (DG) solver for 2D Euler equations. This is based on the matlab code [https://github.com/tcew/nodal-dg](https://github.com/tcew/nodal-dg) It includes modules for mesh generation, boundary condition handling, and numerical flux computation.

## Requirements
To run this project, you need the following Python packages installed:

- `numpy`
- `scipy`

You can install the required packages using the following command:
```bash
pip install numpy scipy
```

## Project Structure
- **`Mesh.py`**: Contains the `Elements` class for mesh generation and geometric computations.
- **`Euler2D.py`**: Implements the 2D Euler solver using the DG method.
- **`euler_BC_IC.py`**: Defines boundary and initial conditions for the Euler equations.
- **`utils.py`**: Utility functions for reading mesh files and comparing results with MATLAB outputs.
- **`constants.py`**: Contains constants used throughout the project.

## Usage

### 1. Test Mesh Generation
To test the mesh generation and geometric computations, run:
```bash
python Mesh.py
```

### 2. Solve 2D Euler Equations
To solve the 2D Euler equations using the DG method, run:
```bash
python Euler2D.py
```

### 3. Compare Results with MATLAB
If you have MATLAB `.mat` files for comparison, ensure they are in the same directory and run the corresponding scripts to compare results.

## Notes
- Ensure the mesh file (e.g., `vortexA04.neu`) is in the same directory as the scripts.
- The project is designed to work with Gambit `.neu` mesh files.

## License
This project is licensed under the MIT License.
