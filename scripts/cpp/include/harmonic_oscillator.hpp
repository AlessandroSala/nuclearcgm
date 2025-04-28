#pragma once
#include "local_potential.hpp"
/**
 * @brief Implements a Harmonic oscillator
 * V(r) = -V0 / (1 + exp((r - R) / diff))
 */
class HarmonicOscillatorPotential : public LocalPotential {
public:
    /**
     * @brief Constructs a  potential term.
     */
    HarmonicOscillatorPotential(double omega_x, double omega_y, double omega_z);

    /**
     * @brief Calculates the Harmonic contribution to H(n0, n1).
     * Only non-zero for diagonal elements (n0 == n1).
     */
    double getValue(double x, double y, double z) const override;

public:
    double omega_x;
    double omega_y;
    double omega_z;
};
