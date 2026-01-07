#pragma once
#include "grid.hpp"
#include "types.hpp"
ComplexDenseMatrix harmonic_oscillator_guess(const Grid &grid, int nev,
                                             double a, double beta,
                                             bool useSpinOrbit = false);
