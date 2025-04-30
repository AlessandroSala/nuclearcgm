#pragma once
#include "local_potential.hpp"
class SphericalCoulombPotential : public LocalPotential {
public:
    /**
     * @brief Constructs a Coulomb potential of a spherical charge distribution.
     * @param R Nuclear radius parameter (e.g., 1.27 * A^(1/3) fm).
     */
    SphericalCoulombPotential(int Z, double R);

    double getValue(double x, double y, double z) const override;

public:
    int Z;
    double R;    // Radius parameter
};
