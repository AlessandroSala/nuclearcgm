[Eigen]: https://libeigen.gitlab.io/
# HF(B) Code
define-name is a 3D lattice, spatial symmetry unrestricted, Skyrme Hartree-Fock(-Bogoliubov) code for the solution of the nuclear many-body problem. The program is written using C++ and open-mpi for multi-threading.

The code allows ground state calculations of even nuclei, with the possibility of constraining spatial deformations.

## Building the code
This repository is equipped with a Makefile for compiling on Linux using `gcc`.
The [Eigen] library is included and used to perform Linear Algebra operations. The LAPACK and BLAS libraries are also linked to speed-up certain operations. If the user wishes to disable this option, the macros
```
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
``` 
and linking in the makefile
```
-lopenblas -llapack
```
should be removed.

## Running 
To run the program, start by executing the output `out.o`. The code will execute every input file in the `input/exec` folder.
### Output
The output will be written in the `output_directory` provided in the input file through an `output_name.txt` summary of the execution and an output folder where fields and data in `JSON` format are saved.
## Input definition
define-name uses `JSON` as its input format. Several examples of possible inputs are provided in the `input/templates` folder.


### Lattice definition
The grid on which the equations are solved is defined by the `box` attribute in the input file.
```
"box": {
    "sideSize": 9.0,
    "axisGridPoints": 30
}
```
The box is then defined on the three x, y, z axes as `[-sideSize, +sideSize]`, with a number of grid points on each axis specified by `axisGridPoints`. Only an even number of points is allowed to prevent the origin singularity in some of the equations.

## Performance and running times
Running times for a single input file greatly vary, depending on: Mass number, Lattice points, Many-body ansatz, i.e. HF, HF+BCS, HFB, Spatial constraints, Symmetry breaking

A step size around 0.4 fm generally provides precision of the total energy in the order of the keV.

## Contact
Report bugs and issues to 
* [Alessandro Sala](mailto:alessandro19.sala@mail.polimi.it)
* [Gianluca Colò](mailto:gianluca.colo@mi.infn.it)

