# DGax: Discontinuous Galerkin Solver for 2D Euler Equations

## Overview
This project implements a Discontinuous Galerkin (DG) solver for 2D Euler equations. This is based on the matlab code [https://github.com/tcew/nodal-dg](https://github.com/tcew/nodal-dg) It includes modules for mesh generation, boundary condition handling, and numerical flux computation.

## Requirements
To run this project, you need the following Python packages installed:

- `recurivenodes`
- `numpy`
- `scipy`

You can install the required packages using the following command:
```bash
pip install numpy recurisivenodes
```

## Package Structure
- **`mesh/`**: Contains the `mesh` and `TriMesh` class for mesh geometric computations.
- **`riemann_solvers/`**: Implements the riemann solvers for DG.
- **`integrators/`**: Defines SSPRK type of integrators.
- **`equations/`**: Equations implemented, currently 2D Euler equations only.
- **`limiters/`**: Limiters for DG Methods.

## Usage

Not yet.

## Notes
- Ensure the mesh file (e.g., `*.neu`) is in the same directory as the scripts.
- The project is designed to work with Gambit `.neu` mesh files.

## License
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg