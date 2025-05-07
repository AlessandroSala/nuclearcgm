#pragma once
#include "types.hpp"
#include "grid.hpp"
ComplexDenseMatrix gaussian_guess(const Grid& grid, int nev, double a);
ComplexDenseMatrix anisotropic_gaussian_guess(const Grid& grid, int nev, double a_x, double a_y, double a_z);