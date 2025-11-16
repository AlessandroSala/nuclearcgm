#pragma once
#include "grid.hpp"
#include "types.hpp"
ComplexDenseMatrix gaussian_guess(const Grid &grid, int nev, double a);
ComplexDenseMatrix harmonic_oscillator_guess(const Grid &grid, int nev,
                                             double a, double beta,
                                             bool useSpinOrbit = false);
ComplexDenseMatrix anisotropic_gaussian_guess(const Grid &grid, int nev,
                                              double a_x, double a_y,
                                              double a_z);
